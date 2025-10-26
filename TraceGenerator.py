"""
TraceGenerator
Generate synthetic network data.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

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

    inter_pkt_time = 1.0  # Average time between packets (seconds per packet)

    seq_length = 128  # Length of each sequence

@dataclass
class TraceData:
    pkt_arrival_times_v:np.ndarray
    inter_pkt_times_v:np.ndarray
    pkt_size_v:np.ndarray
    backlog_v:np.ndarray
    latency_v:np.ndarray
    capacity_v:np.ndarray
    queue_bytes_v:np.ndarray
    dropped_status:np.ndarray
    dropped_sizes:np.ndarray
    dropped_indices:np.ndarray


class TraceGenerator:
    link_properties: LinkProperties = None
    num_training_samples, seq_length_training = 0, 0
    num_val_samples, seq_length_val = 0, 0
    num_test_samples, seq_length_test = 0, 0
    trace_data = []
    dataX_tensor_v, dataY_tensor_v = None, None
    dataX_val_tensor_v, dataY_val_tensor_v = None, None
    dataX_test_tensor_v, dataY_test_tensor_v = None, None
    train_loader:data.DataLoader = None
    val_loader:data.DataLoader = None
    test_loader:data.DataLoader = None

    def __init__(self, link_properties:LinkProperties):
        self.link_properties = link_properties

    def generate_trace_sample(self, seq_length:int):
        # XXX testing random arrival rates to get more diverse training sequences
        trace_data = TraceData
        arrival_rate = np.random.uniform(self.link_properties.min_arrival_rate, self.link_properties.max_arrival_rate)
        inter_pkt_time = 1.0 / arrival_rate
        trace_data.pkt_arrival_times_v = np.cumsum(np.random.exponential(inter_pkt_time, seq_length))
        trace_data.pkt_size_v = np.rint(
            np.random.uniform(self.link_properties.min_pkt_size, self.link_properties.max_pkt_size, seq_length))  # Packet size between 60 and 1000 bytes
        capacity_s = np.random.uniform(self.link_properties.min_capacity, self.link_properties.max_capacity)
        trace_data.capacity_v = np.repeat(capacity_s, seq_length)  # Link capacity (bytes per unit time)
        queue_bytes_s = np.rint(np.random.uniform(self.link_properties.min_queue_bytes, self.link_properties.max_queue_bytes))  # size of queue in bytes
        trace_data.queue_bytes_v = np.repeat(queue_bytes_s, seq_length)
        trace_data.backlog_v = np.zeros(seq_length)
        trace_data.latency_v = np.zeros(seq_length)  # Track latency (proportional to backlog)
        queue = trace_data.pkt_size_v[0]  # Current queue size in bytes
        trace_data.backlog_v[0] = queue
        trace_data.latency_v[0] = queue / capacity_s

        trace_data.dropped_sizes = []  # Store dropped packet sizes
        trace_data.dropped_indices = []  # Store the indices of dropped packets
        trace_data.dropped_status = np.zeros(seq_length, dtype=int)

        for i in range(1, seq_length):
            time_passed = trace_data.pkt_arrival_times_v[i] - trace_data.pkt_arrival_times_v[i - 1]  # Time between packets
            queue = max(0, queue - time_passed * capacity_s)  # Process queued packets

            if queue + trace_data.pkt_size_v[i] > queue_bytes_s:
                trace_data.dropped_sizes.append(trace_data.pkt_size_v[i])  # Store the dropped packet size
                trace_data.dropped_indices.append(i)  # Store the packet index
                trace_data.dropped_status[i] = 1  # this time step drop or not drop
            else:
                queue += trace_data.pkt_size_v[i]  # Accept the packet

            trace_data.backlog_v[i] = queue
            trace_data.latency_v[i] = queue / capacity_s  # Store latency at this time step

        trace_data.inter_pkt_times_v = np.diff(
            np.insert(trace_data.pkt_arrival_times_v, 0, 0))  # shouldnt this just give us back the inter_pkt_time_in?

        return trace_data


    def generate_trace_data_set(self, seq_length, num_samples):
        dataX= []  # Input sequences
        dataY = []  # Target output values
        self.trace_data = []

        for _ in range(num_samples):
            trace_sample = self.generate_trace_sample(seq_length)
            input_features_v = np.stack((trace_sample.inter_pkt_times_v * trace_sample.capacity_v, trace_sample.pkt_size_v, trace_sample.capacity_v, trace_sample.queue_bytes_v),axis=-1)
            output_features_v = np.stack((trace_sample.backlog_v, trace_sample.dropped_status), axis=-1)

            dataX.append(input_features_v)
            dataY.append(output_features_v)
            self.trace_data.append(trace_sample)

        dataX_tensor_v = torch.tensor(np.array(dataX), dtype=torch.float32)
        dataY_tensor_v = torch.tensor(np.array(dataY), dtype=torch.float32)
        print(f"Data shape: X - {dataX_tensor_v.shape}, Y - {dataY_tensor_v.shape}")
        return dataX_tensor_v, dataY_tensor_v


    def create_loaders(self, num_training_samples, seq_length_training, num_val_samples, seq_length_val, num_test_samples, seq_length_test, batch_size=64):
        self.num_training_samples = num_training_samples
        self.seq_length_training = seq_length_training
        dataX_tensor_v, dataY_tensor_v = self.generate_trace_data_set(num_training_samples, seq_length_training)
        print(dataX_tensor_v.shape, dataY_tensor_v.shape)

        self.num_val_samples = num_val_samples  # Number of validation samples
        self.seq_length_val = seq_length_val  # Length of each sequence
        dataX_val_tensor_v, dataY_val_tensor_v = self.generate_trace_data_set(num_val_samples, seq_length_val)
        print(dataX_val_tensor_v.shape, dataY_val_tensor_v.shape)

        self.num_test_samples = num_test_samples
        self.seq_length_test = seq_length_test  # Length of each sequence
        dataX_test_tensor_v, dataY_test_tensor_v = self.generate_trace_data_set(num_test_samples, seq_length_test)
        # print("\nFirst 3 test samples:\n", dataX_test_tensor_v[:1])
        print(dataX_test_tensor_v.shape, dataY_test_tensor_v.shape)

        self.train_loader = data.DataLoader(data.TensorDataset(dataX_tensor_v, dataY_tensor_v), shuffle=True,
                                       batch_size=batch_size)
        self.val_loader = data.DataLoader(data.TensorDataset(dataX_val_tensor_v, dataY_val_tensor_v), shuffle=True,
                                     batch_size=batch_size)
        self.test_loader = data.DataLoader(TensorDataset(dataX_test_tensor_v, dataY_test_tensor_v), shuffle=True,
                                      batch_size=batch_size)

    def plot_inputs(self, index=0):
        plt.rcParams.update({
            'font.size': 20,
            'font.weight': 'bold',
            'axes.labelsize': 22,
            'axes.labelweight': 'bold',
            'axes.titlesize': 22,
            'axes.linewidth': 2.0,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'xtick.major.width': 1.8,
            'ytick.major.width': 1.8,
            'xtick.major.size': 8,
            'ytick.major.size': 8,
            'legend.fontsize': 15,
            'legend.frameon': True,
            'lines.linewidth': 2,
            'pdf.fonttype': 42  # for LaTeX compatibility
        })

        plt.figure(figsize=(12, 6))
        arrival_times_v = self.trace_data[index].pkt_arrival_times_v
        backlog_v = self.trace_data[index].backlog_v
        pkt_size_v = self.trace_data[index].pkt_size_v
        dropped_status = self.trace_data[index].dropped_status
        plt.plot(arrival_times_v, backlog_v, label="Generated Backlog", color='green', linewidth=2.5)
        plt.plot(arrival_times_v, pkt_size_v, label="Packet size", color='red', linestyle="dashed", linewidth=2)
        plt.scatter(arrival_times_v[dropped_status == 1], backlog_v[dropped_status == 1], color='blue', marker='x',
                    label="Dropped Packets", linewidth=2.5)

        plt.xlabel("Time step", fontsize=18, fontweight='bold')
        plt.ylabel("Backlog", fontsize=18, fontweight='bold')
        plt.legend()
        plt.grid(True, linewidth=1.2)
        plt.tight_layout()
        # plt.savefig('b_lim_backlog_new.pdf',format="pdf",  bbox_inches='tight')
        plt.show()



