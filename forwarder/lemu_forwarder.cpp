/*
 * DPDK L2 Forwarding with Timestamping
 *
 * This C++ program implements a high-speed, low-latency packet forwarder
 * between two DPDK-managed ports (A and B).
 *
 * Architecture:
 * Direction A->B: [RX Port A] -> Ring1 -> [Latency Worker] -> Ring2 -> [TX Port B]
 * Direction B->A: [RX Port B] -> Ring3 -> [Latency Worker] -> Ring4 -> [TX Port A]
 *
 * Requirements:
 * - 7 CPU Cores Total (1 Main + 6 Workers)
 * - 2 Dynamic Mbuf Fields (Timestamp, Latency)
 * - 4 rte_rings
 *
 */

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <csignal>
#include <cstdint>
#include <fstream>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_mbuf_dyn.h>
#include <rte_ring.h>
#include <rte_mempool.h>

// The LEmuRnn class for predicting packet handling based on a trained model.
#include "lemurnn.h"

#define BURST_SIZE 32
#define NUM_MBUFS 8191 * 2 // Increased pool size for buffering in 4 rings
#define MBUF_CACHE_SIZE 250
#define RING_SIZE 16384

// RX/TX descriptor defaults
#define RX_DESC_DEFAULT 1024
#define TX_DESC_DEFAULT 1024

// --- Globals ---

// Flag to signal threads to quit
volatile bool force_quit = false;

// Offset for the dynamic mbuf field to store our timestamp
static int timestamp_dynfield_offset = -1;
static int send_time_dynfield_offset = -1;

// this program uses the TS timer, so we need to know how fast it is
static uint64_t tsc_rate_uint64;
static double tsc_rate;

// global info about the prediction model being used
static int hidden_size = 0;
static int num_layers = 0;
static std::string model_type = "rnn";
static std::string model_file;
static double capacity = 1.0;      // units of [Kbit/ms]=[Mbit/s]
static double queue_size = 5.0; // units of [KByte]
std::string data_save_filename_base;

// keep track of some packet stats
uint32_t packet_count = 0;
double packets_total = 0.0;
double packets_dropped = 0.0;
double start_time_ms = 0.0;

// Helper macro to get the timestamp pointer from an mbuf
#define TIMESTAMP_FIELD(mbuf) \
    RTE_MBUF_DYNFIELD((mbuf), timestamp_dynfield_offset, uint64_t*)
#define SEND_TIME_FIELD(mbuf) \
    RTE_MBUF_DYNFIELD((mbuf), send_time_dynfield_offset, uint64_t*)

// --- Signal Handler ---

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        std::cout << "\nSignal " << signum << " received, preparing to exit..." << std::endl;
        force_quit = true;
    }
}

// --- Port Initialization ---

/**
 * @brief Initializes a single DPDK port.
 * Configures 1 RX queue and 1 TX queue.
 *
 * @param port_id The ID of the port to initialize.
 * @param pool The mbuf pool to associate with the port's RX queue.
 */
static void port_init(uint16_t port_id, struct rte_mempool *pool) {
    struct rte_eth_conf port_conf;
    struct rte_eth_dev_info dev_info;
    struct rte_eth_txconf tx_conf;
    struct rte_eth_rxconf rx_conf;
    int ret;

    // Basic port configuration
    memset(&port_conf, 0, sizeof(struct rte_eth_conf));
    
    // Get device info
    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret != 0) {
        throw std::runtime_error("Failed to get device info for port " + std::to_string(port_id));
    }

    // Use default RX/TX configurations
    rx_conf = dev_info.default_rxconf;
    tx_conf = dev_info.default_txconf;

    // Enable basic offloads if supported
    //XXX these macros are deprecated in newer DPDK
    //	  Just leave them out for now
    // https://mails.dpdk.org/archives/dev/2021-November/228464.html
    //if (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_IPV4_CKSUM)
    //    port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_IPV4_CKSUM;
    //if (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_UDP_CKSUM)
    //    port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_UDP_CKSUM;
    //if (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_TCP_CKSUM)
    //    port_conf.rxmode.offloads |= DEV_RX_OFFLOAD_TCP_CKSUM;
	//
    //if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_IPV4_CKSUM)
    //    port_conf.txmode.offloads |= DEV_TX_OFFLOAD_IPV4_CKSUM;
    //if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_UDP_CKSUM)
    //    port_conf.txmode.offloads |= DEV_TX_OFFLOAD_UDP_CKSUM;
    //if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_TCP_CKSUM)
    //    port_conf.txmode.offloads |= DEV_TX_OFFLOAD_TCP_CKSUM;

    // Configure the port with 1 RX queue and 1 TX queue
    ret = rte_eth_dev_configure(port_id, 1, 1, &port_conf);
    if (ret != 0) {
        throw std::runtime_error("rte_eth_dev_configure failed for port " + std::to_string(port_id));
    }

    // Get socket ID for NUMA-aware memory allocation
    int socket_id = rte_eth_dev_socket_id(port_id);
    if (socket_id == SOCKET_ID_ANY) {
        socket_id = 0;
    }

    // Setup RX queue (Queue 0)
    ret = rte_eth_rx_queue_setup(port_id, 0, RX_DESC_DEFAULT, socket_id, &rx_conf, pool);
    if (ret < 0) {
        throw std::runtime_error("rte_eth_rx_queue_setup failed for port " + std::to_string(port_id));
    }

    // Setup TX queue (Queue 0)
    ret = rte_eth_tx_queue_setup(port_id, 0, TX_DESC_DEFAULT, socket_id, &tx_conf);
    if (ret < 0) {
        throw std::runtime_error("rte_eth_tx_queue_setup failed for port " + std::to_string(port_id));
    }

    // Start the port
    ret = rte_eth_dev_start(port_id);
    if (ret < 0) {
        throw std::runtime_error("rte_eth_dev_start failed for port " + std::to_string(port_id));
    }

    // Enable promiscuous mode
    ret = rte_eth_promiscuous_enable(port_id);
    if (ret != 0) {
        throw std::runtime_error("rte_eth_promiscuous_enable failed for port " + std::to_string(port_id));
    }

    std::cout << "Port " << port_id << " initialized successfully." << std::endl;
}

// --- LCore Function Parameters ---

struct rx_lcore_params {
    uint16_t port_id;
    struct rte_ring *ring_out;
};

struct prediction_lcore_params {
    struct rte_ring *ring_in;
    struct rte_ring *ring_out;
    std::string data_save_filename;
    uint16_t rx_port;
};

struct tx_lcore_params {
    uint16_t port_id;
    struct rte_ring *ring_in;
};

// --- LCore Thread Functions ---

/**
 * RX thread
 * - pulls incoming packets from queue
 * - get arrival timestamp and compute inter-packet time
 * - use lemurnn model to predict latency and drop status
 * - discard or record expect TX time and push to latency queue
 */
static int rx_thread_main(void *arg) {
    struct rx_lcore_params *params = (struct rx_lcore_params *)arg;
    uint16_t port_id = params->port_id;
    struct rte_ring *ring = params->ring_out;
    struct rte_mbuf *bufs[BURST_SIZE];
    uint16_t nb_rx, nb_enq, i;

    uint32_t lcore_id = rte_lcore_id();
    std::cout << "Starting RX thread on lcore " << lcore_id << " for port " << port_id << std::endl;

    while (!force_quit) {
        nb_rx = rte_eth_rx_burst(port_id, 0, bufs, BURST_SIZE);
        if (nb_rx == 0) {
            continue;
        }

        // arrival timestamp from TSC clock
        // https://doc.dpdk.org/api-1.6/rte__cycles_8h.html
        uint64_t now = rte_rdtsc();

        // record arrival timestamps
        for (i = 0; i < nb_rx; i++) {
            *TIMESTAMP_FIELD(bufs[i]) = now;
        }
        
        nb_enq = rte_ring_sp_enqueue_burst(ring, (void * const *)bufs, nb_rx, NULL);
        if (unlikely(nb_enq < nb_rx)) {
            rte_pktmbuf_free_bulk(&bufs[nb_enq], nb_rx - nb_enq);
	    std::cout << "WARNING: RX thread dropped packets because rte_ring is full!" << std::endl;
        }
    }

    std::cout << "Exiting RX thread on lcore " << lcore_id << std::endl;
    return 0;
}


/**
 * PREDICTION thread
 * - pulls incoming packets from RX rte_ring
 * - get arrival timestamp and compute inter-packet time
 * - use lemurnn model to predict latency and drop status
 * - discard or record expect TX time and push to latency queue
 */
static int prediction_thread_main(void *arg) {
    struct prediction_lcore_params *params = (struct prediction_lcore_params *)arg;
    struct rte_ring *ring_in = params->ring_in;
    struct rte_ring *ring_out = params->ring_out;
    uint16_t rx_port = params->rx_port;
    std::string data_save_filename = params->data_save_filename;
    struct rte_mbuf *bufs[BURST_SIZE];
    uint16_t nb_dq, nb_enq, i;
    uint64_t prediction_timer = 0;
    double prediction_time_ms = 0.0;
    bool is_lstm = false;

    uint32_t lcore_id = rte_lcore_id();
    std::cout << "Starting PREDICTION thread on lcore " << lcore_id << std::endl;

    // initialize a LEmuRnn model for this thread
    
    if (model_type == "lstm" || model_type == "LSTM") {
        is_lstm = true;
    }
    LEmuRnn model(model_file, num_layers, hidden_size, capacity, queue_size, is_lstm);
    
    // file to write info about packet arrivals and actions
    std::ofstream data_save_file;
    if (!data_save_filename.empty()) {
        data_save_filename += std::to_string(rx_port) + ".dat";
        data_save_file.open(data_save_filename, std::ios::out | std::ios::trunc);
        if (!data_save_file.is_open()) {
            std::cerr << "Error: Could not open file " << data_save_filename << " for writing." << std::endl;
            return 1;
        }
    }
    // need to keep track of the arrival time of the last packet
    uint64_t last_packet_time_tsc = 0;
    uint64_t start_time_tsc = rte_rdtsc();
    
    // this is shared between the threads
    // one will overwrite the other, and it doesn't matter.
    // Just need a common reference point.
    start_time_ms = 1000.0*((double)start_time_tsc)/tsc_rate;
    
    // keep track of packet count
    uint32_t packet_count = 0;
    uint32_t num_drops = 0;

    while (!force_quit) {
        nb_dq = rte_ring_sc_dequeue_burst(ring_in, (void **)bufs, BURST_SIZE, NULL);
        if (nb_dq == 0) continue;
        std::cout << "PREDICTION thread dequeued: " << nb_dq << std::endl;

        for (i = 0; i < nb_dq; i++) {
            uint64_t arrival_tsc = *TIMESTAMP_FIELD(bufs[i]);
            uint16_t size_byte = rte_pktmbuf_pkt_len(bufs[i]);
            double size_kbyte = ((double)size_byte)/1000.0;
            uint64_t inter_packet_time_tsc = arrival_tsc - last_packet_time_tsc;
            double inter_packet_time_ms = 1000.0*((double)inter_packet_time_tsc)/tsc_rate;
            double arrival_ms = 1000.0*((double)(arrival_tsc-start_time_tsc))/tsc_rate;
            uint64_t send_time_tsc = 0;

            // When the program first starts, or if the link has been idle for awhile, 
            // the inter-packet time will be way outside the range the model has been
            // trained for.  In these cases, assume the system is empty anyway and
            // start from zero.
            if (inter_packet_time_ms > 1000.0 * 30.0) {
                packets_total = 0;
                packets_dropped = 0;
                inter_packet_time_ms = 0.0;
                //XXX should we also reset the model hidden state?
            }
            double processed_kbit = inter_packet_time_ms * capacity;

            //std::cout << "packet prediction: " << inter_packet_time_ms
            //     << "\t" << size_kbyte << std::endl;
            //LEmuRnn::PacketAction pa = {1.5, true};

            prediction_timer = rte_rdtsc();
            LEmuRnn::PacketAction pa = model.predict(inter_packet_time_ms, size_kbyte);
            prediction_time_ms = 1000.0*(((double)(rte_rdtsc() - prediction_timer))/tsc_rate);
            std::cout << "\tprediction took ms: " << prediction_time_ms << std::endl;
            packets_total += 1;
            if (pa.drop) {
                packets_dropped += 1;
                num_drops++;  // local version.  Should entfern the other one.
                rte_pktmbuf_free(bufs[i]);
                std::cout << "\tdrop!!" << std::endl;
            } else {
                std::cout << "\tPacket Action: " << pa.latency_ms
                          << "\t" << pa.drop << std::endl;
                send_time_tsc = arrival_tsc + (uint64_t)(pa.latency_ms * tsc_rate / 1000.0);
                //uint64_t send_time_tsc = arrival_tsc + (uint64_t)(pa.latency_ms * 1.0);
                std::cout << arrival_tsc << "\t" << inter_packet_time_tsc  << "\t" << inter_packet_time_ms
                          << "\t" << pa.latency_ms << "\t" << send_time_tsc << std::endl;
                *SEND_TIME_FIELD(bufs[i]) = send_time_tsc;
                nb_enq = rte_ring_sp_enqueue(ring_out, (void *)bufs[i]);
                if (unlikely(nb_enq != 0)) {
                    rte_pktmbuf_free(bufs[i]);
    	            std::cout << "WARNING: PREDICTION thread dropped packets because rte_ring is full!"
    	                      << std::endl;
    	            std::cout << "number enqueued: " << nb_enq << std::endl;
                }
            }
            // at this point we have all the available info except what time the 
            // packet actually gets sent.
            // Doing this in the prediction thread wastes even more time.
            // But it's not much compared to the prediction itself, and all out couts.
            // We'd like the first columns to match the trace files, so we can run
            // arbitrary models against it and verify the results.
            // - packet_number
            // - size [Bytes]
            // - transmit_time [epoch seconds, float]
            // - receive_time [epoch seconds, float, 0 if dropped]
            // - latency [seconds, float]
            // - dropped_status [1 or 0]
            // 
            if (data_save_file.is_open()) {
                data_save_file << packet_count << "\t"
                               << size_byte << "\t"
                               << (arrival_ms/1000.0) << "\t"
                               << ((arrival_ms + pa.latency_ms)/1000.0) << "\t"
                               << (pa.latency_ms/1000.0) << "\t"
                               << pa.drop << "\t"
                               << inter_packet_time_ms << "\t"
                               << processed_kbit << "\t"
                               << size_byte << "\t"
                               << pa.latency_ms << "\t"
                               << prediction_time_ms << "\t"
                               << num_drops << "\t"
                               << arrival_tsc << "\t"
                               << arrival_ms << "\t"
                               << inter_packet_time_tsc << "\t"
                               << size_kbyte << "\t"
                               << send_time_tsc << std::endl;
            }
            packet_count++;
            
            last_packet_time_tsc = arrival_tsc;
        }
    }
    
    if (data_save_file.is_open()) {
        data_save_file.close();
    }

    std::cout << "Exiting PREDICTION thread on lcore " << lcore_id << std::endl;
    return 0;
}

/**
 * TX thread
 * - dequeue packets from the rte_ring
 * - read out their intended send time
 * - busy-wait until the time arrives
 * - send out the packet
 */
static int tx_thread_main(void *arg) {
    struct tx_lcore_params *params = (struct tx_lcore_params *)arg;
    uint16_t port_id = params->port_id;
    struct rte_ring *ring = params->ring_in;
    struct rte_mbuf *bufs[BURST_SIZE];
    uint16_t nb_dq, nb_tx, i;
    uint64_t timestamp, send_time_tsc;
    double wait_ms;
    
    uint32_t lcore_id = rte_lcore_id();
    std::cout << "Starting TX thread on lcore " << lcore_id << " for port " << port_id << std::endl;
    
    uint64_t now = rte_rdtsc();
    while (!force_quit) {
        nb_dq = rte_ring_sc_dequeue_burst(ring, (void **)bufs, BURST_SIZE, NULL);
        if (nb_dq == 0) {
            continue;
        }

	std::cout << "\tdrop rate: " << (packets_dropped/packets_total) << std::endl;

        // using BURST_SIZE>1, we have to loop over the possibly multiple mbufs.
        for (i = 0; i < nb_dq; i++) {
            timestamp = *TIMESTAMP_FIELD(bufs[i]);
            send_time_tsc = *SEND_TIME_FIELD(bufs[i]);
            // busy-wait until it is time to send this one.
	    now = rte_rdtsc();

            if (now < send_time_tsc) {
                wait_ms = 1000.0 * ((double)(send_time_tsc - now))/tsc_rate;
		std::cout << "TX[" << port_id << "]:  waiting tsc: " << (send_time_tsc - now)
			  << "\tms: " << wait_ms << std::endl;
		//std::cout << "\t\tnow:" << now << "\tsend_time: " << send_time_tsc << std::endl;
		//std::cout << "\t\tnow:" << now << "\tsend_time: " << send_time_tsc << std::endl;
	    }
            while (now < send_time_tsc && !force_quit) {
                //std::cout << "now < send_time_tsc\t" << now << "\t" << send_time_tsc << std::endl;
                now = rte_rdtsc();
            }

            // aaaand off she goes...
            nb_tx = rte_eth_tx_burst(port_id, 0, &(bufs[i]), 1);

            // if the mbuf failed to send, free it
            // don't expect this to ever happen            
            if (unlikely(nb_tx != 1)) {
                rte_pktmbuf_free(bufs[i]);
		std::cout << "WARNING: TX thread dropped packets because transmit queue is full!" << std::endl;
            }
        }
    }

    std::cout << "Exiting TX thread on lcore " << lcore_id << std::endl;
    return 0;
}

static void print_usage(char *program_name) {
    std::cerr << "Usage: " << program_name
	      << " -h <hidden_size> -l <layers> -m <model_file> -c <capacity(Mb/s)> -q <queuesize(KB)> -t <model_type>\n";
}

/**
 * Method to parse the command line args.
 * I'm sloppy here and just store them in global vars declared at the top.
 */
int parse_options(int argc, char *argv[]) {

    int opt;
    while ((opt = getopt(argc, argv, "h:l:m:c:q:t:f:")) != -1) {
        switch (opt) {
        case 'h':
            try {
                // optarg is a global char* pointer set by getopt containing the value
                hidden_size = std::stoi(optarg);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: -h requires a valid integer.\n";
                return 1;
            }
            break;
        case 'l':
            try {
                num_layers = std::stoi(optarg);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: -l requires a valid integer.\n";
                return 1;
            }
            break;
        case 'c':
            try {
                capacity = std::stod(optarg);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: -c requires a valid double.\n";
                return 1;
            }
            break;
        case 'q':
            try {
                queue_size = std::stod(optarg);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: -q requires a valid double.\n";
                return 1;
            }
            break;
        case 'm':
            model_file = optarg;
            break;
        case 't':
            model_type = optarg;
            break;
        case 'f':
            data_save_filename_base = optarg;
            break;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validation check
    if (hidden_size == 0 || num_layers == 0 || model_file.empty()) {
        std::cerr << "Error: Missing required arguments.\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "--- POSIX Getopt Parsed ---\n";
    std::cout << "Hidden Size: " << hidden_size << "\n";
    std::cout << "Num Layers:  " << num_layers << "\n";
    std::cout << "Model File:  " << model_file << "\n";
    std::cout << "Capacity:    " << capacity << "\n";
    std::cout << "Queue size:  " << queue_size << "\n";
    std::cout << "Model type:  " << model_type << "\n";
    std::cout << "Data save file:  " << data_save_filename_base << "\n";

    return 0;
}


/**
 * main()
 */
int main(int argc, char *argv[]) {
    int ret;
    uint16_t port_a, port_b;
    uint16_t nb_ports;
    uint32_t lcore_id;
    struct rte_mempool *mbuf_pool;
    struct rte_ring *ring_rx_to_prediction_ab, *ring_prediction_to_tx_ab;
    struct rte_ring *ring_rx_to_prediction_ba, *ring_prediction_to_tx_ba;

    // dpdk gets to parse the args before the "--"
    ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");
    }
    argc -= ret;
    argv += ret;

    // next parse the port/device numbers
    if (argc < 3) {
        rte_exit(EXIT_FAILURE, "Usage: %s [EAL_ARGS] <PORT_A_ID> <PORT_B_ID>\n", argv[0]);
    }
    try {
        port_a = (uint16_t)std::stoul(argv[1]);
        port_b = (uint16_t)std::stoul(argv[2]);
        argc -= 2;
        argv += 2;
    } catch (const std::exception &e) {
        rte_exit(EXIT_FAILURE, "Invalid port ID argument: %s\n", e.what());
    }

    // finally we parse the prediction model options
    if (parse_options(argc, argv)) {
        std::cerr << "ERROR: command line parsing error." << std::endl;
        return 1;
    }
    
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    if (rte_lcore_count() < 5) { // 1 main + 4 workers
        rte_exit(EXIT_FAILURE, "This application requires at least 4 worker lcores (5 total).\n");
    }

    nb_ports = rte_eth_dev_count_avail();
    if (port_a >= nb_ports || port_b >= nb_ports) {
        rte_exit(EXIT_FAILURE, "Invalid port ID. Available ports: %u\n", nb_ports);
    }

    // create the mbuf pool
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (mbuf_pool == NULL) {
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool: %s\n", rte_strerror(rte_errno));
    }

    // register the dynamic mbuf field for the timestamp
    static const struct rte_mbuf_dynfield timestamp_dynfield_desc = {
        .name = "l2fwd_timestamp_field",
        .size = sizeof(uint64_t),
        .align = __alignof(uint64_t),
    };
    timestamp_dynfield_offset = rte_mbuf_dynfield_register(&timestamp_dynfield_desc);
    if (timestamp_dynfield_offset < 0) {
        rte_exit(EXIT_FAILURE, "Cannot register mbuf dynfield: %s\n", rte_strerror(rte_errno));
    }
    static const struct rte_mbuf_dynfield send_time_dynfield_desc = {
        .name = "l2fwd_send_time_field",
        .size = sizeof(uint64_t),
        .align = __alignof(uint64_t),
    };
    send_time_dynfield_offset = rte_mbuf_dynfield_register(&send_time_dynfield_desc);
    if (send_time_dynfield_offset < 0) {
        rte_exit(EXIT_FAILURE, "Cannot register mbuf dynfield: %s\n", rte_strerror(rte_errno));
    }

    // initialize ports
    try {
        port_init(port_a, mbuf_pool);
        port_init(port_b, mbuf_pool);
    } catch (const std::exception &e) {
        rte_exit(EXIT_FAILURE, "Port initialization failed: %s\n", e.what());
    }

    // create the four rte_rings
    ring_rx_to_prediction_ab = rte_ring_create("RING_RX2P_A2B", RING_SIZE, rte_socket_id(),
                                  RING_F_SP_ENQ | RING_F_SC_DEQ);
    ring_prediction_to_tx_ab = rte_ring_create("RING_P2TX_A2B", RING_SIZE, rte_socket_id(),
                                  RING_F_SP_ENQ | RING_F_SC_DEQ);
    ring_rx_to_prediction_ba = rte_ring_create("RING_RX2P_B2A", RING_SIZE, rte_socket_id(),
                                  RING_F_SP_ENQ | RING_F_SC_DEQ);
    ring_prediction_to_tx_ba = rte_ring_create("RING_P2TX_B2A", RING_SIZE, rte_socket_id(),
                                  RING_F_SP_ENQ | RING_F_SC_DEQ);
    
    if (ring_rx_to_prediction_ab == NULL || ring_prediction_to_tx_ab == NULL
        || ring_rx_to_prediction_ba == NULL || ring_prediction_to_tx_ba == NULL) {
        rte_exit(EXIT_FAILURE, "Cannot create rings: %s\n", rte_strerror(rte_errno));
    }

    // assign tasks to lcores
    std::vector<uint32_t> worker_lcores;
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        worker_lcores.push_back(lcore_id);
    }
    if (worker_lcores.size() < 4) {
        rte_exit(EXIT_FAILURE, "Could not find 4 worker lcores. Found %zu.\n", worker_lcores.size());
    }

    static struct rx_lcore_params rx_params_ab, rx_params_ba;
    static struct prediction_lcore_params prediction_params_ab, prediction_params_ba;
    static struct tx_lcore_params tx_params_ab, tx_params_ba;

    // flow: A -> B
    rx_params_ab = {port_a, ring_rx_to_prediction_ab};
    prediction_params_ab = {ring_rx_to_prediction_ab, ring_prediction_to_tx_ab, data_save_filename_base, port_a};
    tx_params_ab = {port_b, ring_prediction_to_tx_ab};

    // flow: B -> A
    rx_params_ba = {port_b, ring_rx_to_prediction_ba};
    prediction_params_ba = {ring_rx_to_prediction_ba, ring_prediction_to_tx_ba, data_save_filename_base, port_b};
    tx_params_ba = {port_a, ring_prediction_to_tx_ba};

    std::cout << "Main lcore " << rte_lcore_id() << " launching threads..." << std::endl;
    std::cout << "  lcore " << worker_lcores[0] << ": RX Port A (" << port_a << ") -> Ring A->B" << std::endl;
    std::cout << "  lcore " << worker_lcores[1] << ": Ring A->B -> TX Port B (" << port_b << ")" << std::endl;
    std::cout << "  lcore " << worker_lcores[2] << ": RX Port B (" << port_b << ") -> Ring B->A" << std::endl;
    std::cout << "  lcore " << worker_lcores[3] << ": Ring B->A -> TX Port A (" << port_a << ")" << std::endl;

    // find out what our time units are
    tsc_rate_uint64 = rte_get_timer_hz();	
    tsc_rate = (double)tsc_rate_uint64;
    std::cout << "TSC rate: " << tsc_rate_uint64 << "\t" << tsc_rate << std::endl;

    // launch threads
    rte_eal_remote_launch(rx_thread_main, &rx_params_ab, worker_lcores[0]);
    rte_eal_remote_launch(prediction_thread_main, &prediction_params_ab, worker_lcores[1]);   
    rte_eal_remote_launch(tx_thread_main, &tx_params_ab, worker_lcores[2]);
    rte_eal_remote_launch(rx_thread_main, &rx_params_ba, worker_lcores[3]);
    rte_eal_remote_launch(prediction_thread_main, &prediction_params_ba, worker_lcores[4]);   
    rte_eal_remote_launch(tx_thread_main, &tx_params_ba, worker_lcores[5]);

    // wait for all threads to exit
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (rte_eal_wait_lcore(lcore_id) < 0) {
            std::cerr << "Error waiting for lcore " << lcore_id << std::endl;
        }
    }

    std::cout << "Shutting down..." << std::endl;
    rte_eth_dev_stop(port_a);
    rte_eth_dev_close(port_a);
    rte_eth_dev_stop(port_b);
    rte_eth_dev_close(port_b);

    // Note: Rings and mempools are automatically freed on EAL cleanup
    // rte_ring_free(ring_a_to_b);
    // rte_ring_free(ring_b_to_a);
    // rte_mempool_free(mbuf_pool);

    rte_eal_cleanup();
    std::cout << "Done." << std::endl;

    return 0;
}
