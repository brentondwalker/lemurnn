"""
TrafficGeneratorBurstyCBR
Constant bit rate traffic in bursts.
Allows for random packet sizes, and adjusts the packet rate depending on each packet's size.
This needs parameters for the transitions between burst and off.  We have not set up a way to
configure that yet, so I'll hard-code it for now.  Adjust the intensity of the on periods
to achieve the desired expected total rate.
"""

import numpy as np
from LinkProperties import LinkProperties
from TrafficGenerator import TrafficGenerator, PacketInfo


class TrafficGeneratorBurstyCBR(TrafficGenerator, traffic_type='bursty_cbr'):

    # Since this is CBR, make the burst lengths constant for each sample.
    # The interval between bursts will be relative to the packet rate.
    burst_length_low = 2
    burst_length_high = 64

    def __init__(self, link_properties:LinkProperties):
        super().__init__(link_properties)
        # [Kbit/ms]
        self.arrival_rate = np.random.uniform(link_properties.min_arrival_rate, link_properties.max_arrival_rate)
        self.mean_pkt_size_kbyte = (link_properties.max_pkt_size + link_properties.min_pkt_size) / 2
        self.burst_length = np.random.randint(self.burst_length_low, self.burst_length_high+1)
        self.arrival_rate_burst = self.arrival_rate * self.burst_length
        self.inter_burst_time_ms = (1.0 / self.arrival_rate) * (self.burst_length - 1)
        self.burst_counter = 0

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
        if self.burst_counter == self.burst_length:
            ipt = self.inter_burst_time_ms
            self.burst_counter = 0
        else:
            ipt = pkt_size_kbyte * 8 / self.arrival_rate / self.burst_length
        self.last_pkt_time_ms += ipt
        self.byte_count += pkt_size_kbyte * 1000
        self.packet_count += 1
        self.burst_counter += 1
        return PacketInfo(pkt_size_kbyte, self.last_pkt_time_ms, ipt)
