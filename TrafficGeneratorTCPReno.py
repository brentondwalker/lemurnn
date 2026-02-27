""""
TrafficGeneratorTCPReno
Generates a stream of traffic with AIMD congestion control.
This is not have completely accurate TCP behavior because it finds out immediately the latency and dropped status
of the previous packet.  IRL TCP has to wait one RTT for feedback on the latency, and slightly more for loss detection.
- Right now this class will send data at the max sustainable rate.
- We will treat CWND as being in units of segments, not bytes.
- Slow start is not implemented yet.
"""

import numpy as np
from LinkProperties import LinkProperties
from TrafficGenerator import TrafficGenerator, PacketInfo


class TrafficGeneratorTCPReno(TrafficGenerator, traffic_type='tcp_reno'):

    # the alpha and beta parameters for AIMD
    alpha = 1.0
    beta = 0.5

    def __init__(self, link_properties:LinkProperties):
        super().__init__(link_properties)
        # [Kbit/ms]
        self.arrival_rate = np.random.uniform(link_properties.min_arrival_rate, link_properties.max_arrival_rate)
        self.mean_pkt_size_kbyte = (link_properties.max_pkt_size + link_properties.min_pkt_size) / 2
        self.cwnd = 1.0


    def next_packet(self, last_latency:float=0.0, last_drop:bool=False):
        """
        Return the next PacketInfo
        :param last_latency:
        :param last_drop:
        :return:
        """
        if last_drop:
            self.cwnd *= (1.0 - self.beta)
        else:
            self.cwnd += 1.0 / self.cwnd

        # given the current cwnd and RTT, what rate can we send at (segments per time)?
        # assume RTT = last_latency * 2
        # so the packet rate is CWND/RTT
        ipt = last_latency / self.cwnd

        # size of next packet is random from an interval
        pkt_size_kbyte = np.random.uniform(self.link_properties.min_pkt_size, self.link_properties.max_pkt_size+1)

        self.last_pkt_time_ms += ipt
        self.byte_count += pkt_size_kbyte * 1000
        self.packet_count += 1
        return PacketInfo(pkt_size_kbyte, self.last_pkt_time_ms, ipt)
