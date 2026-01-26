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


class TraceGeneratorPacketQueue(TraceGenerator):

    def __init__(self, link_properties:list[LinkProperties], input_str='bscq', output_str='bd', normalize=False):
        super().__init__(link_properties, input_str, output_str)
        self.data_type = 'packetqueue'
        self.normalize = normalize

    def get_extra_dataset_properties(self):
        """
        This will be called by save_dataset_properties() in the superclass.
        """
        extra_dataset_properties = {
        }
        return extra_dataset_properties

    def generate_trace_sample(self, lp:LinkProperties, seq_length:int):
        arrival_rate = np.random.uniform(lp.min_arrival_rate, lp.max_arrival_rate)
        inter_pkt_time = 1.0 / arrival_rate
        pkt_arrival_times_v = np.cumsum(np.random.exponential(inter_pkt_time, seq_length))
        pkt_size_v = np.rint(
            np.random.uniform(lp.min_pkt_size, lp.max_pkt_size, seq_length))  # Packet size between 60 and 1000 bytes
        capacity_s = np.random.uniform(lp.min_capacity, lp.max_capacity)
        capacity_v = np.repeat(capacity_s, seq_length)  # Link capacity (bytes per unit time)

        # we'll derive queue a queue size range by dividing the queue range in bytes by the max and min packet size
        min_queue_size = int(lp.min_queue_bytes/lp.max_pkt_size)
        max_queue_size = int(lp.max_queue_bytes/lp.min_pkt_size)

        # queue size of <= 0 indicates infinite queue
        if lp.max_queue_bytes <= 0:
            # but we dont want to pass inf as one of the inputs, so set that to zero
            queue_size_s = np.inf
            queue_size_v = np.repeat(0, seq_length)
        else:
            queue_size_s = np.rint(np.random.uniform(min_queue_size, max_queue_size))  # size of queue in bytes
            queue_size_v = np.repeat(queue_size_s, seq_length)
        backlog_v = np.zeros(seq_length)
        latency_v = np.zeros(seq_length)  # Track latency (proportional to backlog)
        queue_bytes = pkt_size_v[0]  # Current queue length in bytes
        queue_pkts = deque([pkt_size_v[0]])
        total_bytes_sent = 0

        backlog_v[0] = queue_bytes
        latency_v[0] = backlog_v[0] * 8 / capacity_s

        dropped_sizes = []  # Store dropped packet sizes
        dropped_indices = []  # Store the indices of dropped packets
        dropped_status = np.zeros(seq_length, dtype=int)

        for i in range(1, seq_length):
            time_passed = pkt_arrival_times_v[i] - pkt_arrival_times_v[i - 1]  # Time between packets
            bytes_processed_interval = time_passed * capacity_s
            queue_bytes = max(0, queue_bytes - bytes_processed_interval)  # Process queued packets
            # go through the queue and figure out which packets departed
            total_bytes_sent += bytes_processed_interval
            while len(queue_pkts) > 0 and total_bytes_sent >= queue_pkts[0]:
                total_bytes_sent -= queue_pkts[0]
                queue_pkts.popleft()

            if len(queue_pkts) == 0:
                total_bytes_sent = 0

            if len(queue_pkts) + 1 > queue_size_s:
                dropped_sizes.append(pkt_size_v[i])  # Store the dropped packet size
                dropped_indices.append(i)  # Store the packet index
                dropped_status[i] = 1  # this time step drop or not drop
            else:
                queue_pkts.append(pkt_size_v[i])  # Accept the packet
                queue_bytes += pkt_size_v[i]

            backlog_v[i] = queue_bytes
            latency_v[i] = backlog_v[i] * 8 / capacity_s  # Store latency at this time step

        inter_pkt_times_v = np.diff(
            np.insert(pkt_arrival_times_v, 0, 0))  # shouldnt this just give us back the inter_pkt_time_in?

        if self.normalize:
            print("WARNING: normalization no longer supported in this branch")

        return TraceSample(pkt_arrival_times_v, inter_pkt_times_v, pkt_size_v, backlog_v,
                            latency_v, capacity_v, queue_size_v, dropped_status,
                            dropped_sizes, dropped_indices)



