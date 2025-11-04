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

link_properties_library = {
    'default':      LinkProperties(min_arrival_rate=0.2,
                                   max_arrival_rate=1,
                                   min_capacity=500,
                                   max_capacity=1000,
                                   min_pkt_size=500,
                                   max_pkt_size=1500,
                                   min_queue_bytes=2500,
                                   max_queue_bytes=10000),

    'inifinite_queue': LinkProperties(min_arrival_rate=0.2,
                              max_arrival_rate=1,
                              min_capacity=500,
                              max_capacity=1000,
                              min_pkt_size=500,
                              max_pkt_size=1500,
                              min_queue_bytes=0,
                              max_queue_bytes=0),

    '1-10Mbps':     LinkProperties(min_arrival_rate=125,  # 125 pkt/s ~~ 1Mbps
                                   max_arrival_rate=1250, # 1250 pkt/s ~~ 10Mbps
                                   min_capacity = 6.25e5, # 6.25e5 B/s = 5 Mbps
                                   max_capacity = 1.5e6,  # 1.5e6 B/s = 12 Mbps
                                   min_pkt_size=500,
                                   max_pkt_size=1500,
                                   min_queue_bytes=1500,
                                   max_queue_bytes=100*1000) # 100 average-sized packets
}