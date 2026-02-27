"""
TrafficGeneratorBurstyExponential
Generate synthetic network traffic with exponential inter-packet times sent in bursts.
Use a 2-state CTMC to control bursty periods.
"""

import numpy as np
from LinkProperties import LinkProperties, link_properties_library
from TrafficGenerator import TrafficGenerator, PacketInfo


class TrafficGeneratorBurstyExponential(TrafficGenerator, traffic_type='bursty_exponential'):

    # The interval between bursts will be relative to the packet rate.
    burst_length_low = 2
    burst_length_high = 64

    def __init__(self, link_properties:LinkProperties):
        super().__init__(link_properties)
        self.traffic_type = 'bursty_exponential'
        # [Kbit/ms]
        self.arrival_rate_kbit_ms = np.random.uniform(self.link_properties.min_arrival_rate, self.link_properties.max_arrival_rate)
        self.mean_pkt_size_kbyte = (self.link_properties.max_pkt_size + self.link_properties.min_pkt_size)/2.0
        self.burst_length = np.random.randint(self.burst_length_low, self.burst_length_high+1)

        self.pkt_arrival_rate_ms = self.arrival_rate_kbit_ms / (8 * self.mean_pkt_size_kbyte)
        self.mean_inter_packet_time_ms = 1.0 / self.pkt_arrival_rate_ms

        # arrival_rate = np.random.uniform(lp.min_arrival_rate, lp.max_arrival_rate)
        # mean_pkt_size_kbyte = (lp.max_pkt_size + lp.min_pkt_size) / 2

        # pkt_arrival_rate_ms = arrival_rate / (8 * mean_pkt_size_kbyte)
        # inter_pkt_time_ms = 1.0 / pkt_arrival_rate_ms   # [ms/pkt]
        # pkt_arrival_times_v = np.cumsum(np.random.exponential(inter_pkt_time_ms, seq_length))

    def next_packet(self, last_latency:float=0.0, last_drop:bool=False):
        """
        Return the next PacketInfo
        :param last_latency:
        :param last_drop:
        :return:
        """
        pkt_size_kbyte = np.random.uniform(self.link_properties.min_pkt_size, self.link_properties.max_pkt_size+1)
        ipt = np.random.exponential(self.mean_inter_packet_time_ms)
        self.last_pkt_time_ms += ipt
        self.byte_count += pkt_size_kbyte * 1000
        self.packet_count += 1
        return PacketInfo(pkt_size_kbyte, self.last_pkt_time_ms, ipt)

# main() for testing
def main():
    tg = TrafficGenerator.create(link_properties_library['default'])
    for i in range(0,10):
        print(i, tg.next_packet())

# ======================================
# ======================================
# ======================================

if __name__ == "__main__":
    main()
