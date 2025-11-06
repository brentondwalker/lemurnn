"""
TraceGenerator
Generate synthetic network data.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

from TraceGenerator import TraceGenerator, LinkProperties, TraceSample


class TraceGeneratorCodel(TraceGenerator):

    def __init__(self, link_properties: LinkProperties, input_str='bscq', output_str='bd', normalize=False, base_interval=10, codel_threshold=5):
        super().__init__(link_properties, input_str, output_str)
        self.base_interval = base_interval
        self.codel_threshold = codel_threshold
        self.interval_denominator = 1
        self.data_type = 'codel'
        self.normalize = normalize


    def get_extra_dataset_properties(self):
        """
        This will be called by save_dataset_properties() in the superclass.
        """
        extra_dataset_properties = {
            'base_interval': self.base_interval,
            'codel_threshold': self.codel_threshold
        }
        return extra_dataset_properties

    def generate_trace_sample(self, seq_length:int):
        base_interval = 10
        interval_denominator = 1
        codel_threshold = 5
        current_time = 0.0
        interval_start = 0.0
        interval_end = interval_start + base_interval / np.sqrt(interval_denominator)
        interval_min_latency = 0

        arrival_rate = np.random.uniform(self.link_properties.min_arrival_rate, self.link_properties.max_arrival_rate)
        inter_pkt_time = 1.0 / arrival_rate
        pkt_arrival_times_v = np.cumsum(np.random.exponential(inter_pkt_time, seq_length))
        pkt_size_v = np.rint(
            np.random.uniform(self.link_properties.min_pkt_size, self.link_properties.max_pkt_size, seq_length))  # Packet size between 60 and 1000 bytes
        capacity_s = np.random.uniform(self.link_properties.min_capacity, self.link_properties.max_capacity)
        capacity_v = np.repeat(capacity_s, seq_length)  # Link capacity (bytes per unit time)
        queue_bytes_s = np.rint(np.random.uniform(self.link_properties.min_queue_bytes, self.link_properties.max_queue_bytes))  # size of queue in bytes
        queue_bytes_v = np.repeat(queue_bytes_s, seq_length)
        backlog_v = np.zeros(seq_length)
        latency_v = np.zeros(seq_length)  # Track latency (proportional to backlog)
        queue = pkt_size_v[0]  # Current queue size in bytes
        backlog_v[0] = queue
        latency_v[0] = queue / capacity_s

        dropped_sizes = []  # Store dropped packet sizes
        dropped_indices = []  # Store the indices of dropped packets
        dropped_status = np.zeros(seq_length, dtype=int)

        for i in range(1, seq_length):
            time_passed = pkt_arrival_times_v[i] - pkt_arrival_times_v[i - 1]  # Time between packets
            current_time += time_passed

            if current_time >= interval_end:
                interval_start = current_time
                # print("\tmin_latency: ", interval_min_latency)
                if interval_min_latency > codel_threshold:
                    dropped_sizes.append(pkt_size_v[i])  # Store the dropped packet size
                    dropped_indices.append(i)  # Store the packet index
                    dropped_status[i] = 1  # this time step drop or not drop
                    interval_denominator += 1
                else:
                    interval_denominator = 1

                interval_end = interval_start + base_interval / np.sqrt(interval_denominator)
                # print("new interval: ", interval_start, interval_end, interval_denominator)
                interval_min_latency = 0.0

            queue = max(0, queue - time_passed * capacity_v[i])  # Process queued packets

            if not dropped_status[i]:
                queue += pkt_size_v[i]  # Accept the packet

            pkt_latency = queue / capacity_v[i]
            if (interval_min_latency == 0) or (pkt_latency < interval_min_latency):
                interval_min_latency = pkt_latency

            backlog_v[i] = queue  # Store backlog at this time step
            latency_v[i] = queue / capacity_s  # Store latency at this time step

        inter_pkt_times_v = np.diff(np.insert(pkt_arrival_times_v, 0, 0))

        if self.normalize:
            backlog_v /= self.link_properties.max_pkt_size
            pkt_size_v /= self.link_properties.max_pkt_size
            capacity_v /= self.link_properties.max_pkt_size
            queue_bytes_v /= self.link_properties.max_pkt_size

        return TraceSample(pkt_arrival_times_v, inter_pkt_times_v, pkt_size_v, backlog_v,
                            latency_v, capacity_v, queue_bytes_v, dropped_status,
                            dropped_sizes, dropped_indices)



