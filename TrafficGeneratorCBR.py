"""
TrafficGeneratorCBR
Constant bit rate traffic.
Allows for random packet sizes, and adjusts the packet rate depending on each packet's size.
"""

import numpy as np
from LinkProperties import LinkProperties
from TrafficGenerator import TrafficGenerator, PacketInfo


class TrafficGeneratorCBR(TrafficGenerator, traffic_type='cbr'):

    traffic_type = 'cbr'

    def __init__(self, link_properties:LinkProperties):
        super().__init__(link_properties)
        # [Kbit/ms]
        self.arrival_rate = np.random.uniform(link_properties.min_arrival_rate, link_properties.max_arrival_rate)
        self.mean_pkt_size_kbyte = (link_properties.max_pkt_size + link_properties.min_pkt_size) / 2

    # mean_pkt_size_kbyte = (lp.max_pkt_size + lp.min_pkt_size) / 2

    # pkt_arrival_rate_ms = arrival_rate / (8 * mean_pkt_size_kbyte)
    # inter_pkt_time_ms = 1.0 / pkt_arrival_rate_ms   # [ms/pkt]

    def next_packet(self, last_latency:float=0.0, last_drop:bool=False):
        """
        Return the next PacketInfo
        :param last_latency:
        :param last_drop:
        :return:
        """
        # size of next packet is random from an interval
        pkt_size_kbyte = np.random.uniform(self.link_properties.min_pkt_size, self.link_properties.max_pkt_size+1)
        # compute what the inter-packet time would be for a packet of this size
        # [ms] = [KByte][8 bit/Byte]/[Kbit/ms]
        ipt = pkt_size_kbyte * 8 / self.arrival_rate
        self.last_pkt_time_ms += ipt
        self.byte_count += pkt_size_kbyte * 1000
        self.packet_count += 1
        return PacketInfo(pkt_size_kbyte, self.last_pkt_time_ms, ipt)
