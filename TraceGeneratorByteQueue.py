"""
TraceGenerator
Generate synthetic network data.
"""
import json
import random

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

    def __init__(self, link_properties:list[LinkProperties], input_str='bscq', output_str='bd', normalize=False):
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

    def generate_trace_sample(self, lp:LinkProperties, seq_length:int):
        # compute the packet arrival rate [pkt/ms] to achieve the desired arrival rate
        arrival_rate = np.random.uniform(lp.min_arrival_rate, lp.max_arrival_rate)
        mean_pkt_size_kbyte = (lp.max_pkt_size + lp.min_pkt_size) / 2
        # [Kbit/ms] / [bit/Byte][KByte/pkt] = [pkt/ms]
        pkt_arrival_rate_ms = arrival_rate / (8 * mean_pkt_size_kbyte)
        inter_pkt_time_ms = 1.0 / pkt_arrival_rate_ms   # [ms/pkt]
        pkt_arrival_times_v = np.cumsum(np.random.exponential(inter_pkt_time_ms, seq_length))
        # if we measure packet size in KByte, then we can't round the size to integer values
        pkt_size_v = np.random.uniform(lp.min_pkt_size, lp.max_pkt_size, seq_length)
        capacity_s = np.random.uniform(lp.min_capacity, lp.max_capacity)
        capacity_v = np.repeat(capacity_s, seq_length)  # Link capacity (bytes per unit time)
        # queue size of <= 0 indicates infinite queue
        if lp.max_queue_bytes <= 0:
            # but we don't want to pass inf as one of the inputs, so set that to zero
            queue_size_s = np.inf
            queue_size_v = np.repeat(0, seq_length)
        else:
            queue_size_s = np.rint(np.random.uniform(lp.min_queue_bytes, lp.max_queue_bytes))
            queue_size_v = np.repeat(queue_size_s, seq_length)

        # In order to model the per-packet overhead of ethernet (or other networks)
        # we need to track how many packets are in the queue
        # These overhead bytes take up time on the link, but not space in the queue
        queued_packets = 0

        # In case an inter-packet interval leaves us in the middle of a packet overhead
        remaining_overhead_kbytes = 0

        backlog_v = np.zeros(seq_length)
        latency_v = np.zeros(seq_length)  # Track latency (proportional to backlog)
        queue_kbytes = pkt_size_v[0]  # Current queue length in kbytes
        queue_pkts = deque([pkt_size_v[0]])
        total_kbytes_sent = 0

        backlog_v[0] = queue_kbytes + lp.overhead_bytes
        latency_v[0] = backlog_v[0] * 8 / capacity_s   # [KByte]*[bit/Byte]/[KBit/ms] = [ms]

        dropped_sizes = []  # Store dropped packet sizes
        dropped_indices = []  # Store the indices of dropped packets
        dropped_status = np.zeros(seq_length, dtype=int)

        for i in range(1, seq_length):
            time_passed_ms = pkt_arrival_times_v[i] - pkt_arrival_times_v[i - 1]  # Time between packets
            kbytes_processed_interval = time_passed_ms * capacity_s / 8
            queue_kbytes = max(0, queue_kbytes - kbytes_processed_interval)  # Process queued packets
            # go through the queue and figure out which packets departed
            total_kbytes_sent += kbytes_processed_interval

            # a lot of logic required for the possibility that a packet arrival happens
            # during the TX of the overhead between packets
            if total_kbytes_sent >= remaining_overhead_kbytes:
                total_kbytes_sent -= remaining_overhead_kbytes
                remaining_overhead_kbytes = 0
            else:
                remaining_overhead_kbytes -= total_kbytes_sent
                total_kbytes_sent = 0

            while len(queue_pkts) > 0 and total_kbytes_sent >= queue_pkts[0]:
                total_kbytes_sent -= queue_pkts[0]
                queue_pkts.popleft()
                # see how much of the packet overhead we also sent
                if total_kbytes_sent >= lp.overhead_bytes:
                    remaining_overhead_kbytes = 0
                else:
                    remaining_overhead_kbytes = lp.overhead_bytes - total_kbytes_sent
                    total_kbytes_sent = 0

            if len(queue_pkts) == 0 and remaining_overhead_kbytes == 0:
                total_kbytes_sent = 0

            #if len(queue_pkts) + 1 > queue_size_s:
            if queue_kbytes + pkt_size_v[i] > queue_size_s:
                dropped_sizes.append(pkt_size_v[i])  # Store the dropped packet size
                dropped_indices.append(i)  # Store the packet index
                dropped_status[i] = 1  # this time step drop or not drop
            else:
                queue_pkts.append(pkt_size_v[i])  # Accept the packet
                queue_kbytes += pkt_size_v[i]

            backlog_v[i] = queue_kbytes + lp.overhead_bytes * len(queue_pkts) + remaining_overhead_kbytes
            latency_v[i] = backlog_v[i] * 8 / capacity_s   # [KByte][bit/Byte]/[KBit/ms] = [ms]

        inter_pkt_times_v = np.diff(
            np.insert(pkt_arrival_times_v, 0, 0))  # shouldn't this just give us back the inter_pkt_time_in?

        if self.normalize:
            print("WARNING: normalization no longer supported in this branch")

        return TraceSample(pkt_arrival_times_v, inter_pkt_times_v, pkt_size_v, backlog_v,
                            latency_v, capacity_v, queue_size_v, dropped_status,
                            dropped_sizes, dropped_indices)



