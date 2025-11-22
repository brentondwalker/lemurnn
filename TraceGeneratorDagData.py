"""
TraceGenerator
Generate synthetic network data.
"""
import csv
import json
import os
import re
import sys

import numpy as np
from collections import deque
import glob
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

from TraceGenerator import TraceGenerator, LinkProperties, TraceSample


class TraceGeneratorDagData(TraceGenerator):

    def __init__(self, link_properties:LinkProperties=None, input_str='bscq', output_str='bd', normalize=False, datadirs=None):
        super().__init__(link_properties, input_str, output_str)
        self.data_type = 'dag'
        self.normalize = normalize
        self.datadirs = datadirs

        # make an index of all the sample files, and
        self.sample_files = []
        print(f"iterating over {datadirs}")
        for path in self.datadirs:
            print(f"looking for csv files in: {path}")
            if os.path.isfile(path):
                print()
                self.sample_files += [path]
            else:
                self.sample_files += (glob.glob(f"{path}/**/mgtrace_C*_L*_Q*_*_*.csv", recursive=True))
        print(f"TraceGeneratorDagData: identified {len(self.sample_files)} samples")
        self.sample_sequence = np.random.permutation(len(self.sample_files))
        self.sample_index = 0


    def get_extra_dataset_properties(self):
        """
        This will be called by save_dataset_properties() in the superclass.
        """
        extra_dataset_properties = {
            'datadirs': self.datadirs
        }
        return extra_dataset_properties


    def parse_filename(self, filename):
        """
        Parses the filename to extract experiment parameters.
        Expected format: mgtrace_C<C>_L<L>_Q<Q>_..._<exp_num>.csv
        Example: "mgtrace_C9_L0_Q9800_1762957671_7.csv"
        """
        # Regex to find C, L, and Q parameters
        # It looks for C, L, and Q followed by digits
        match = re.search(r'C(\d+)_L(\d+)_Q(\d+)', filename)

        if match:
            c_val = float(match.group(1))
            l_val = float(match.group(2))
            q_val = float(match.group(3))
            return c_val, l_val, q_val
        else:
            print(f"Warning: Could not parse parameters from filename: {filename}", file=sys.stderr)
            return None, None, None


    def generate_trace_sample(self, seq_length:int):
        """
        As a sort of normalization, we try to express everything in KByte and KByte/ms.
        This should keep input and output values from getting to ridiculous.
        """
        if self.sample_index >= len(self.sample_files):
            self.sample_sequence = np.random.permutation(len(self.sample_files))
            self.sample_index = 0
            print(f"WARNING: TraceGeneratorDagData: not enough samples!  Recycling the pool.")

        # randomly take the next sample file
        # the next one that has at least seq_length data points
        first_sample_index = self.sample_index
        num_rows = 0
        while num_rows-1 < seq_length:
            sample_filename = self.sample_files[self.sample_sequence[self.sample_index]]
            with open(f"{sample_filename}", 'r', newline='') as csvfile:
                num_rows = sum(1 for row in csvfile)
            self.sample_index += 1
            if first_sample_index == self.sample_index:
                print(f"ERROR: no files in the list are as long as {seq_length}")
                sys.exit(0)

        c_val, l_val, q_val = self.parse_filename(sample_filename)
        # the CAP value is Mbit/s = Kbit/ms.  No need to convert.
        capacity_s = np.float32(c_val)

        # queue size is in Bytes, so convert to KByte
        queue_size_s = np.float32(q_val) / 1000

        inter_pkt_times_v = np.zeros(seq_length)
        pkt_size_v = np.zeros(seq_length)
        latency_v = np.zeros(seq_length)
        capacity_v = np.zeros(seq_length)
        queue_size_v = np.zeros(seq_length)
        dropped_status = np.zeros(seq_length)
        dropped_sizes = []
        dropped_indices = []

        try:
            with open(f"{sample_filename}", 'r', newline='') as csvfile:
                # Use tab delimiter as specified
                reader = csv.DictReader(csvfile, delimiter='\t')
                last_tx_time = 0.0
                last_latency = 0.0

                for i, row in enumerate(reader):
                    if i >= seq_length:
                        break
                    capacity_v[i] = capacity_s
                    queue_size_v[i] = queue_size_s
                    try:
                        # 1. Read data from CSV
                        # pkt_size is in Byte, so convert to KByte
                        pkt_size_v[i] = np.float32(row['size']) / 1000
                        dropped_status[i] = np.float32(row['dropped_status'])
                        dropped_sizes.append(int(row['size'])) #XXX maybe convert this to Kbyte too?
                        dropped_indices.append(i)

                        # 2. Calculate 't' (inter-packet time)
                        tx_time = np.float64(row['transmit_time'])
                        if i == 0:
                            inter_pkt_times_v[i] = 0.0  # No inter-packet time for the first packet
                        else:
                            # times are in s, so convert to ms
                            inter_pkt_times_v[i] = 1000 * (tx_time - last_tx_time)
                        last_tx_time = tx_time

                        if dropped_status[i]:
                            latency_v[i] = last_latency
                        else:
                            # latency is in s, so convert to ms
                            latency_v[i] = 1000 * np.float32(row['latency'])
                        last_latency = latency_v[i]

                        # 3. Calculate (pseudo) bits processed (inter-packet time * capacity)
                        #b = t * c_val

                    except (ValueError, KeyError) as e:
                        print(f"Warning: Skipping row {i + 1} in {sample_filename}. Error: {e}", file=sys.stderr)
                        continue

        except IOError as e:
            print(f"Error: Could not read file {sample_filename}. Reason: {e}", file=sys.stderr)
            return None
        except csv.Error as e:
            print(f"Error: CSV parsing error in {sample_filename}. Reason: {e}", file=sys.stderr)
            return None

        pkt_arrival_times_v = np.cumsum(inter_pkt_times_v)
        backlog_v = capacity_s * latency_v

        ts = TraceSample(pkt_arrival_times_v, inter_pkt_times_v, pkt_size_v, backlog_v,
                            latency_v, capacity_v, queue_size_v, dropped_status,
                            dropped_sizes, dropped_indices)
        #print(ts)
        return ts

