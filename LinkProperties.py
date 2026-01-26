from dataclasses import dataclass


@dataclass
class LinkProperties:
    min_arrival_rate:float    # KBit/ms = MBit/s
    max_arrival_rate:float    # KBit/ms = MBit/s
    min_capacity: float       # KBit/ms = MBit/s
    max_capacity: float       # KBit/ms = MBit/s
    min_pkt_size: float       # KByte
    max_pkt_size: float       # KByte
    min_queue_bytes: float    # KByte
    max_queue_bytes: float    # KByte
    overhead_bytes: float     # KByte
    #inter_pkt_time:float = 1.0  # Average time between packets (seconds per packet)
    #seq_length:int = 128  # Length of each sequence

    def infinite_queue(self):
        self.min_queue_bytes = self.max_queue_bytes = 0

link_properties_library = {
    'default':      LinkProperties(min_arrival_rate=0.2,
                                   max_arrival_rate=15,
                                   min_capacity=5,
                                   max_capacity=10,
                                   min_pkt_size=0.06,
                                   max_pkt_size=1.4,
                                   min_queue_bytes=5,
                                   max_queue_bytes=10,
                                   overhead_bytes=0),

    'daglike': LinkProperties(min_arrival_rate=0.2,
                              max_arrival_rate=15,
                              min_capacity=1,
                              max_capacity=10,
                              min_pkt_size=0.06,
                              max_pkt_size=1.4,
                              min_queue_bytes=5,
                              max_queue_bytes=50,
                              overhead_bytes=0.024),  # 24 Bytes of overhead

    'daglike-ping': LinkProperties(min_arrival_rate=0.001,  # 1 pkt/second
                              max_arrival_rate=0.001,
                              min_capacity=1,
                              max_capacity=10,
                              min_pkt_size=0.084,  # 84 bytes
                              max_pkt_size=0.084,
                              min_queue_bytes=5,
                              max_queue_bytes=50,
                              overhead_bytes=0.024),  # 24 Bytes of overhead

    'daglike-overload': LinkProperties(min_arrival_rate=0.2,
                              max_arrival_rate=100,
                              min_capacity=1,
                              max_capacity=10,
                              min_pkt_size=0.06,
                              max_pkt_size=1.4,
                              min_queue_bytes=5,
                              max_queue_bytes=100,
                              overhead_bytes=0.024),  # 24 Bytes of overhead

    'daglike-vping': LinkProperties(min_arrival_rate=0.001,  # 1 pkt/second
                                   max_arrival_rate=0.001,
                                   min_capacity=1,
                                   max_capacity=10,
                                   min_pkt_size=0.084,  # 84 bytes
                                   max_pkt_size=1.4,
                                   min_queue_bytes=5,
                                   max_queue_bytes=50,
                                   overhead_bytes=0.024),  # 24 Bytes of overhead

    # same scale as default, but with less tendency to produce huge backlogs
    'default-light': LinkProperties(min_arrival_rate=0.2,
                              max_arrival_rate=1,
                              min_capacity=1000,
                              max_capacity=1200,
                              min_pkt_size=500,
                              max_pkt_size=1500,
                              min_queue_bytes=2500,
                              max_queue_bytes=10000,
                              overhead_bytes=0),

    'blank': LinkProperties(min_arrival_rate=0.0,
                              max_arrival_rate=0,
                              min_capacity=0,
                              max_capacity=0,
                              min_pkt_size=0,
                              max_pkt_size=0,
                              min_queue_bytes=0,
                              max_queue_bytes=0,
                              overhead_bytes=0),

    # constant capacity, packet size, arrival rate
    'const-rates':      LinkProperties(min_arrival_rate=1,
                                   max_arrival_rate=1,
                                   min_capacity=1000,
                                   max_capacity=1000,
                                   min_pkt_size=900,
                                   max_pkt_size=900,
                                   min_queue_bytes=0,
                                   max_queue_bytes=0,
                                   overhead_bytes=0),

    '1-10Mbps':     LinkProperties(min_arrival_rate=125,  # 125 pkt/s ~~ 1Mbps
                                   max_arrival_rate=1250, # 1250 pkt/s ~~ 10Mbps
                                   min_capacity = 6.25e5, # 6.25e5 B/s = 5 Mbps
                                   max_capacity = 1.5e6,  # 1.5e6 B/s = 12 Mbps
                                   min_pkt_size=500,
                                   max_pkt_size=1500,
                                   min_queue_bytes=5*1000,
                                   max_queue_bytes=10*1000,
                                   overhead_bytes=0), # 10 average-sized packets

    '1-10Mbps-scaled': LinkProperties(min_arrival_rate=125/1000,  # 0.125 kpkt/sec
                               max_arrival_rate=1250/1000,  # 1.25 kpkt/sec
                               min_capacity=6.25e5/1000,  # 625 kB/sec
                               max_capacity=1.5e6/1000,  # 1500 kB/sec
                               min_pkt_size=500,
                               max_pkt_size=1500,
                               min_queue_bytes=5 * 1000,
                               max_queue_bytes=10 * 1000,
                               overhead_bytes=0),  # 10 average-sized packets

    '1Mbps-exact': LinkProperties(min_arrival_rate=125,  # 125 pkt/s ~~ 1Mbps
                               max_arrival_rate=125,  # 1250 pkt/s ~~ 10Mbps
                               min_capacity=6.25e5,  # 6.25e5 B/s = 5 Mbps
                               max_capacity=6.25e5,  # 1.5e6 B/s = 12 Mbps
                               min_pkt_size=500,
                               max_pkt_size=1500,
                               min_queue_bytes=5 * 1000,
                               max_queue_bytes=10 * 1000,
                               overhead_bytes=0),  # 10 average-sized packets

    'experiment': LinkProperties(min_arrival_rate=1.25,   # 1 Mbps
                                  max_arrival_rate=12.5,  # 12.5 pkt/ms = 1250 pkt/s ~~ 10Mbps
                                  min_capacity=1,  # 1 Kb/ms
                                  max_capacity=10,   # 10 Kb/ms
                                  min_pkt_size=0.500,  # 0.5 KB = 500 B
                                  max_pkt_size=1.500,# 1.5 Kb = 1500 B
                                  min_queue_bytes=5, # 5 KB
                                  max_queue_bytes=50,
                                  overhead_bytes=0),# 50 average-sized packets

    # same as experiment but wider range of rates and capacities and pkt sizes
    'experiment2': LinkProperties(min_arrival_rate=0.001,  # 0.001pkt/ms = 1 pkt/s ~~ ping
                                 max_arrival_rate=12.5,  # 12.5 pkt/ms = 1250 pkt/s ~~ 10Mbps
                                 min_capacity=0.5,  # 0.5 Kb/ms
                                 max_capacity=20,  # 20 Kb/ms
                                 min_pkt_size=0.1,  # 0.1 KB = 100 B
                                 max_pkt_size=1.500,  # 1.5 Kb = 1500 B
                                 min_queue_bytes=5,  # 5 KB
                                 max_queue_bytes=50,
                                 overhead_bytes=0),  # 50 average-sized packets

}