from dataclasses import dataclass


@dataclass
class LinkProperties:
    min_arrival_rate:float
    max_arrival_rate:float
    min_capacity: float
    max_capacity: float
    min_pkt_size: int
    max_pkt_size: int
    min_queue_bytes: int
    max_queue_bytes:int
    #inter_pkt_time:float = 1.0  # Average time between packets (seconds per packet)
    #seq_length:int = 128  # Length of each sequence

    def infinite_queue(self):
        self.min_queue_bytes = self.max_queue_bytes = 0

link_properties_library = {
    'default':      LinkProperties(min_arrival_rate=0.2,
                                   max_arrival_rate=1,
                                   min_capacity=500,
                                   max_capacity=1000,
                                   min_pkt_size=500,
                                   max_pkt_size=1500,
                                   min_queue_bytes=2500,
                                   max_queue_bytes=10000),

    # same scale as default, but with less tendency to produce huge backlogs
    'default-light': LinkProperties(min_arrival_rate=0.2,
                              max_arrival_rate=1,
                              min_capacity=1000,
                              max_capacity=1200,
                              min_pkt_size=500,
                              max_pkt_size=1500,
                              min_queue_bytes=2500,
                              max_queue_bytes=10000),

    'blank': LinkProperties(min_arrival_rate=0.0,
                              max_arrival_rate=0,
                              min_capacity=0,
                              max_capacity=0,
                              min_pkt_size=0,
                              max_pkt_size=0,
                              min_queue_bytes=0,
                              max_queue_bytes=0),

    # constant capacity, packet size, arrival rate
    'const-rates':      LinkProperties(min_arrival_rate=1,
                                   max_arrival_rate=1,
                                   min_capacity=1000,
                                   max_capacity=1000,
                                   min_pkt_size=900,
                                   max_pkt_size=900,
                                   min_queue_bytes=0,
                                   max_queue_bytes=0),

    '1-10Mbps':     LinkProperties(min_arrival_rate=125,  # 125 pkt/s ~~ 1Mbps
                                   max_arrival_rate=1250, # 1250 pkt/s ~~ 10Mbps
                                   min_capacity = 6.25e5, # 6.25e5 B/s = 5 Mbps
                                   max_capacity = 1.5e6,  # 1.5e6 B/s = 12 Mbps
                                   min_pkt_size=500,
                                   max_pkt_size=1500,
                                   min_queue_bytes=5*1000,
                                   max_queue_bytes=10*1000), # 10 average-sized packets

    '1-10Mbps-scaled': LinkProperties(min_arrival_rate=125/1000,  # 0.125 kpkt/sec
                               max_arrival_rate=1250/1000,  # 1.25 kpkt/sec
                               min_capacity=6.25e5/1000,  # 625 kB/sec
                               max_capacity=1.5e6/1000,  # 1500 kB/sec
                               min_pkt_size=500,
                               max_pkt_size=1500,
                               min_queue_bytes=5 * 1000,
                               max_queue_bytes=10 * 1000),  # 10 average-sized packets

    '1Mbps-exact': LinkProperties(min_arrival_rate=125,  # 125 pkt/s ~~ 1Mbps
                               max_arrival_rate=125,  # 1250 pkt/s ~~ 10Mbps
                               min_capacity=6.25e5,  # 6.25e5 B/s = 5 Mbps
                               max_capacity=6.25e5,  # 1.5e6 B/s = 12 Mbps
                               min_pkt_size=500,
                               max_pkt_size=1500,
                               min_queue_bytes=5 * 1000,
                               max_queue_bytes=10 * 1000),  # 10 average-sized packets

}