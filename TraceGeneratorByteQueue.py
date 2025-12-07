"""
TraceGenerator
Generate synthetic network data.
"""
import json

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

from TraceGenerator import TraceGenerator, LinkProperties, TraceSample


class TraceGeneratorByteQueue(TraceGenerator):

    def __init__(self, link_properties: LinkProperties, input_str='bscq', output_str='bd', normalize=False):
        super().__init__(link_properties, input_str, output_str)
        self.data_type = 'bytequeue'
        self.normalize = normalize

    def get_extra_dataset_properties(self):
        """
        This will be called by save_dataset_properties() in the superclass.
        """
        extra_dataset_properties = {
        }
        return extra_dataset_properties

    def generate_trace_sample(self, seq_length:int):
        arrival_rate = np.random.uniform(self.link_properties.min_arrival_rate, self.link_properties.max_arrival_rate)
        inter_pkt_time = 1.0 / arrival_rate
        pkt_arrival_times_v = np.cumsum(np.random.exponential(inter_pkt_time, seq_length))
        # if we measure packet size in KByte, then we can't round the size to integer values
        pkt_size_v = np.random.uniform(self.link_properties.min_pkt_size, self.link_properties.max_pkt_size, seq_length)
        capacity_s = np.random.uniform(self.link_properties.min_capacity, self.link_properties.max_capacity)
        capacity_v = np.repeat(capacity_s, seq_length)  # Link capacity (bytes per unit time)
        # queue size of <= 0 indicates infinite queue
        if self.link_properties.max_queue_bytes <= 0:
            # but we don't want to pass inf as one of the inputs, so set that to zero
            queue_size_s = np.inf
            queue_size_v = np.repeat(0, seq_length)
        else:
            queue_size_s = np.rint(np.random.uniform(self.link_properties.min_queue_bytes,
                                                      self.link_properties.max_queue_bytes))  # size of queue in bytes
            queue_size_v = np.repeat(queue_size_s, seq_length)

        # In order to model the per-packet overhead of ethernet (or other networks)
        # we need to track how many packets are in the queue
        # These overhead bytes take up time on the link, but not space in the queue
        queued_packets = 0

        # In case an inter-packet interval leaves us in the middle of a packet overhead
        remaining_overhead_bytes = 0

        backlog_v = np.zeros(seq_length)
        latency_v = np.zeros(seq_length)  # Track latency (proportional to backlog)
        queue_bytes = pkt_size_v[0]  # Current queue length in bytes
        queue_pkts = deque([pkt_size_v[0]])
        total_bytes_sent = 0

        backlog_v[0] = queue_bytes + self.link_properties.overhead_bytes
        latency_v[0] = backlog_v[0] * 8 / capacity_s   # [KByte]*[bit/Byte]/[KBit/ms] = [ms]

        dropped_sizes = []  # Store dropped packet sizes
        dropped_indices = []  # Store the indices of dropped packets
        dropped_status = np.zeros(seq_length, dtype=int)

        for i in range(1, seq_length):
            time_passed = pkt_arrival_times_v[i] - pkt_arrival_times_v[i - 1]  # Time between packets
            bytes_processed_interval = time_passed * capacity_s
            queue_bytes = max(0, queue_bytes - bytes_processed_interval)  # Process queued packets
            # go through the queue and figure out which packets departed
            total_bytes_sent += bytes_processed_interval

            # a lot of logic required for the possibility that a packet arrival happens
            # during the TX of the overhead between packets
            if total_bytes_sent >= remaining_overhead_bytes:
                total_bytes_sent -= remaining_overhead_bytes
                remaining_overhead_bytes = 0
            else:
                remaining_overhead_bytes -= total_bytes_sent
                total_bytes_sent = 0

            while len(queue_pkts) > 0 and total_bytes_sent >= queue_pkts[0]:
                total_bytes_sent -= queue_pkts[0]
                queue_pkts.popleft()
                # see how much of the packet overhead we also sent
                if total_bytes_sent >= self.link_properties.overhead_bytes:
                    remaining_overhead_bytes = 0
                else:
                    remaining_overhead_bytes = self.link_properties.overhead_bytes - total_bytes_sent
                    total_bytes_sent = 0

            if len(queue_pkts) == 0 and remaining_overhead_bytes == 0:
                total_bytes_sent = 0

            if remaining_overhead_bytes > 0:
                print(f"REMAINING_OVERHEAD_BYTES = {remaining_overhead_bytes}")

            #if len(queue_pkts) + 1 > queue_size_s:
            if queue_bytes + pkt_size_v[i] > queue_size_s:
                dropped_sizes.append(pkt_size_v[i])  # Store the dropped packet size
                dropped_indices.append(i)  # Store the packet index
                dropped_status[i] = 1  # this time step drop or not drop
            else:
                queue_pkts.append(pkt_size_v[i])  # Accept the packet
                queue_bytes += pkt_size_v[i]

            backlog_v[i] = queue_bytes + self.link_properties.overhead_bytes * len(queue_pkts) + remaining_overhead_bytes
            latency_v[i] = backlog_v[i] * 8 / capacity_s  # Store latency at this time step

        inter_pkt_times_v = np.diff(
            np.insert(pkt_arrival_times_v, 0, 0))  # shouldnt this just give us back the inter_pkt_time_in?

        if self.normalize:
            print("WARNING: normalization no longer supported in this branch")

        return TraceSample(pkt_arrival_times_v, inter_pkt_times_v, pkt_size_v, backlog_v,
                            latency_v, capacity_v, queue_size_v, dropped_status,
                            dropped_sizes, dropped_indices)



