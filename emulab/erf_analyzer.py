#!/usr/bin/env python3

import json
from collections import namedtuple
import sys
import hashlib
import argparse

# Try to import pyshark
try:
    import pyshark
except ImportError:
    print("Error: 'pyshark' library not found. Please install it with 'pip install pyshark'", file=sys.stderr)
    sys.exit(1)

# Define the structure for the resulting analysis
PacketRecord = namedtuple('PacketRecord', [
    'packet_number',
    'size',
    'transmit_time',
    'receive_time',
    'latency',
    'dropped_status'
])

# Interface definitions for the A--B--C path we are interested in
TX_INTERFACE = 2  # A -> B (Transmit time)
RX_INTERFACE = 1  # B -> C (Receive time)


def compute_latencies(tx_packets:dict, rx_packets:dict):

    print(f"Found {len(tx_packets)} unique packets transmitted (A->B).")
    print(f"Found {len(rx_packets)} unique packets received (B->C).")

    # 2. Second Pass: Analyze and compile the final records
    analysis_records = []
    packet_counter = 1

    # Iterate through all transmitted packets
    for p_id, tx_data in tx_packets.items():
        size = tx_data['size']
        tx_time = tx_data['tx_time']

        # Check if the packet was successfully observed on the RX link
        if p_id in rx_packets:
            rx_data = rx_packets[p_id]
            rx_time = rx_data['rx_time']

            # Successful transmission (Dropped Status = 0)
            latency = rx_time - tx_time
            dropped_status = 0

            # Ensure latency is non-negative (due to clock sync issues or simulation order)
            # If clocks are perfectly synchronized, this ensures correctness.
            latency = max(0.0, latency)

        else:
            # Packet was dropped at node B (Dropped Status = 1)
            rx_time = 0.0
            latency = 0.0
            dropped_status = 1

        # Create the final record
        record = PacketRecord(
            packet_number=packet_counter,
            size=size,
            transmit_time=tx_time,
            receive_time=rx_time,
            latency=latency,
            dropped_status=dropped_status
        )
        analysis_records.append(record)
        packet_counter += 1

    return analysis_records


def analyze_packet_trace(trace_data):
    """
    Processes a list of packet events from a simulated ERF trace to determine
    latency and dropped status for the A -> B -> C path.

    Args:
        trace_data (list): A list of dictionaries, where each dictionary
                           represents a single packet observation event.

    Returns:
        list: A list of PacketRecord named tuples with the analysis results.
    """
    # Dictionary to store packets observed on the TX link (A->B)
    # Key: Unique packet hash
    # Value: { 'size': int, 'tx_time': float }
    tx_packets = {}

    # Dictionary to store packets observed on the RX link (B->C)
    # Key: Unique packet hash
    # Value: { 'rx_time': float }
    rx_packets = {}

    experiment_traces = []

    print(f"--- Processing {len(trace_data)} Packet Observation Events ---")

    # The capture is actually a series of experiments with 5s pauses between them
    # Therefore, we keep track of the last tx timestamp seen.  If we see a gap
    # of more than 4s, we dump the current results and reinitialize the data structures.
    last_tx_time = 0.0

    # 1. First Pass: Separate and record TX and RX events
    for event in trace_data:
        interface = event['interface']
        packet_id = event['packet_id']
        timestamp = event['timestamp']
        size = event['size']

        # check if this is the start of the next experiment
        if interface == TX_INTERFACE:
            if (timestamp - last_tx_time) > 4.0:
                latency_results = compute_latencies(tx_packets, rx_packets)
                print_results(latency_results)
                experiment_traces.append(latency_results)
                tx_packets = {}
                rx_packets = {}

            # Record the transmit event (A->B)
            if packet_id in tx_packets:
                # Handle a potential rare retransmission scenario, we only care about the first one.
                continue
            tx_packets[packet_id] = {
                'size': size,
                'tx_time': timestamp
            }
            last_tx_time = timestamp

        elif interface == RX_INTERFACE:
            # Record the receive event (B->C)
            if packet_id in rx_packets:
                # Handle a potential rare duplicate capture, we only care about the first one.
                continue
            rx_packets[packet_id] = {
                'rx_time': timestamp
            }

        # Interfaces 3 (C->B) and 4 (B->A) are ignored for this specific A->B->C analysis.
    return experiment_traces

def print_results(records):
    """Prints the final analysis table."""
    print("\n--- FINAL A->B->C ANALYSIS REPORT ---")
    print("-" * 75)
    print(
        f"{'#':<4} | {'Size (B)':<8} | {'TX Time (I1)':<15} | {'RX Time (I2)':<15} | {'Latency (s)':<11} | {'Dropped':<7}")
    print("-" * 75)

    for r in records:
        rx_time_str = f"{r.receive_time:.6f}" if r.receive_time > 0 else "N/A"
        latency_str = f"{r.latency:.6f}" if r.latency > 0 else "N/A"
        dropped_str = "YES" if r.dropped_status == 1 else "NO"

        print(
            f"{r.packet_number:<4} | {r.size:<8} | {r.transmit_time:.6f} | {rx_time_str:<15} | {latency_str:<11} | {dropped_str:<7}"
        )
    print("-" * 75)
    print(f"Total packets analyzed: {len(records)}")
    dropped_count = sum(r.dropped_status for r in records)
    print(f"Total packets dropped (A->B): {dropped_count}")


def get_packet_identifier(pkt):
    """
    Returns the IP Identification (ip.id) field as a string.
    The user specified all packets are UDP/IP, so this field should be present
    and stable across the two capture points (A->B and B->C).
    """
    try:
        if 'IP' in pkt:
            # pyshark provides ip.id as a string (e.g., '0x1234' or '4660')
            return pkt.ip.id
    except (AttributeError, TypeError):
        # This can happen if field is missing or packet is not IP
        pass

    # If not an IP packet, we can't get the ID.
    return None


def load_trace_from_erf(filepath):
    """
    Loads an ERF file using pyshark (tshark) and converts it into
    the 'trace_data' list format our analyzer function expects.
    """
    print(f"--- Loading ERF file: {filepath} ---")
    print("This may take a while for large files...")

    trace_data = []
    packet_count = 0
    processed_count = 0

    try:
        # FileCapture uses tshark to read the file
        cap = pyshark.FileCapture(filepath)

        for pkt in cap:
            packet_count += 1
            try:
                #print(f"trying packet {packet_count}")
                # ERF header contains the interface
                if not hasattr(pkt, 'erf'):
                    # Skip packets that don't have an ERF header
                    #print(f"  no header!!")
                    continue
                #print("got a header!!")
                #print(f"  erf: {type(pkt.erf)}\n  {pkt.erf}  ")
                #print(f"  flags: {pkt.erf.flags}\t{type(pkt.erf.flags)}")
                #print(f"  {pkt.erf._all_fields}")
                #print(f"  \nCAP: {int(pkt.erf._all_fields['erf.flags.cap'])}")
                #print(f"  contains erf keys: {pkt.erf}")

                # pyshark provides the 'iface' field from the ERF header
                # It might be a hex string (e.g., '0x1'), so use int(x, 0)
                #print(f"iface: {pkt.erf._all_fields['erf.flags.cap']}")
                interface = int(pkt.erf._all_fields['erf.flags.cap'], 0)
                #print(f"header found  interface {interface}")

                # We only care about interfaces 0, 1, 2, 3
                if interface not in [0, 1, 2, 3]:
                    continue

                # Get timestamp (as float) and size
                timestamp = float(pkt.sniff_timestamp)
                size = int(pkt.length)

                # Get the unique packet identifier (IP.ID)
                packet_id = get_packet_identifier(pkt)

                if packet_id:
                    event = {
                        'interface': interface,
                        'timestamp': timestamp,
                        'size': size,
                        'packet_id': packet_id
                    }
                    trace_data.append(event)
                    processed_count += 1

                if packet_count % 10000 == 0:
                    print(f"  ... scanned {packet_count} packets ...")

            except (AttributeError, TypeError, ValueError) as e:
                # Handle packets tshark might not fully dissect (e.g., non-IP)
                # or issues with field extraction
                print(f"Warning: Skipping packet {packet_count}. Error: {e}", file=sys.stderr)
                pass

        cap.close()

    except pyshark.shark.SharkNotInstalled:
        print("--- ERROR ---", file=sys.stderr)
        print("TShark (part of Wireshark) is not installed or not in your system's PATH.", file=sys.stderr)
        print("Please install Wireshark from https://www.wireshark.org/ and try again.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"--- ERROR: File not found: {filepath} ---", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"--- ERF Load Complete ---")
    print(f"Total packets scanned: {packet_count}")
    print(f"Total packets processed (with valid IP.ID): {processed_count}")
    return trace_data


# --- SIMULATED ERF TRACE DATA ---
# This is no longer used, as we will load from a file.
# SIMULATED_TRACE = [ ... ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Analyze A->B->C packet latency and drops from an ERF trace file."
    )
    parser.add_argument(
        "erf_file",
        help="Path to the .erf packet capture file."
    )
    args = parser.parse_args()

    # 1. Load the trace data from the ERF file
    trace_data = load_trace_from_erf(args.erf_file)

    if not trace_data:
        print("No processable packets found in the ERF file. Exiting.")
        sys.exit(0)

    # 2. Run the analysis (this function is unchanged)
    results = analyze_packet_trace(trace_data)

    print(f"Extracted {len(results)} experiments.")

    #if not results:
    #    print("Analysis complete, but no matching packets were found.")
    #else:
    #    # 3. Print the final results
    #    print_results(results)

    # Example of how you could save the data to a CSV file (recommended for analysis)
    # import csv
    # with open('analysis_report.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(PacketRecord._fields)
    #     writer.writerows(results)
    # print("\nResults also saved to analysis_report.csv")

