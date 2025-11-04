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
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
from LinkProperties import LinkProperties


@dataclass
class TraceSample:
    pkt_arrival_times_v:np.ndarray
    inter_pkt_times_v:np.ndarray
    pkt_size_v:np.ndarray
    backlog_v:np.ndarray
    latency_v:np.ndarray
    capacity_v:np.ndarray
    queue_bytes_v:np.ndarray
    dropped_status:np.ndarray
    dropped_sizes:List[int]
    dropped_indices:List[int]

class TraceGenerator:
    data_type = 'bytequeue'

    def __init__(self, link_properties:LinkProperties, input_str='bscq', output_str='bd'):
        self.link_properties = link_properties
        self.input_str = input_str
        self.output_str = output_str
        self.data_type = 'bytequeue'
        self.seed = None
        self.test_seed = None
        self.num_training_samples, self.seq_length_training = 0, 0
        self.num_val_samples, self.seq_length_val = 0, 0
        self.num_test_samples, self.seq_length_test = 0, 0
        self.trace_data = {}
        self.dataX_tensor_v, self.dataY_tensor_v = None, None
        self.dataX_val_tensor_v, self.dataY_val_tensor_v = None, None
        self.dataX_test_tensor_v, self.dataY_test_tensor_v = None, None
        self.loaders = {
            'train': {},
            'val':   {},
            'test':  {}
        }
        #self.train_loader: data.DataLoader = None
        #self.val_loader: data.DataLoader = None
        #self.test_loader: data.DataLoader = None

    def save_dataset_properties(self, filename):
        dataset_properties = {
            'name': str(self.__class__),
            'data_type': self.data_type,
            'num_training_samples': self.num_training_samples,
            'num_val_samples': self.num_val_samples,
            'num_test_samples': self.num_test_samples,
            'seq_length_training': self.seq_length_training,
            'seq_length_val': self.seq_length_val,
            'seq_length_test': self.seq_length_test,
            'input_str': self.input_str,
            'output_str': self.output_str,
            'input_size': self.input_size(),
            'output_size': self.output_size(),
            'seed': self.seed,
        }
        # add any attributes that are specific to subclasses
        dataset_properties = dict(list(dataset_properties.items()) + list(self.get_extra_dataset_properties().items()))
        with open(filename, "w") as dataset_properties_file:
                dataset_properties_file.write(json.dumps(dataset_properties))
                dataset_properties_file.write("\n")

    def get_extra_dataset_properties(self):
        extra_dataset_properties = {}
        return extra_dataset_properties

    def input_size(self):
        return len(self.input_str)

    def output_size(self):
        return len(self.output_str)


    def generate_trace_sample(self, seq_length:int):
        # XXX testing random arrival rates to get more diverse training sequences
        arrival_rate = np.random.uniform(self.link_properties.min_arrival_rate, self.link_properties.max_arrival_rate)
        inter_pkt_time = 1.0 / arrival_rate
        pkt_arrival_times_v = np.cumsum(np.random.exponential(inter_pkt_time, seq_length))
        pkt_size_v = np.rint(
            np.random.uniform(self.link_properties.min_pkt_size, self.link_properties.max_pkt_size, seq_length))  # Packet size between 60 and 1000 bytes
        capacity_s = np.random.uniform(self.link_properties.min_capacity, self.link_properties.max_capacity)
        capacity_v = np.repeat(capacity_s, seq_length)  # Link capacity (bytes per unit time)

        # queue size of <= 0 indicates infinite queue
        if self.link_properties.max_queue_bytes <= 0:
            # but we dont want to pass inf as one of the inputs, so set that to zero
            queue_bytes_s = np.inf
            queue_bytes_v = np.repeat(0, seq_length)
        else:
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
            queue = max(0, queue - time_passed * capacity_s)  # Process queued packets

            if queue + pkt_size_v[i] > queue_bytes_s:
                dropped_sizes.append(pkt_size_v[i])  # Store the dropped packet size
                dropped_indices.append(i)  # Store the packet index
                dropped_status[i] = 1  # this time step drop or not drop
            else:
                queue += pkt_size_v[i]  # Accept the packet

            backlog_v[i] = queue
            latency_v[i] = queue / capacity_s  # Store latency at this time step

        inter_pkt_times_v = np.diff(
            np.insert(pkt_arrival_times_v, 0, 0))  # shouldnt this just give us back the inter_pkt_time_in?

        return TraceSample(pkt_arrival_times_v, inter_pkt_times_v, pkt_size_v, backlog_v,
                            latency_v, capacity_v, queue_bytes_v, dropped_status,
                            dropped_sizes, dropped_indices)


    def feature_vector_from_sample(self, trace_sample:TraceSample, input_str=None, output_str=None):
        if not input_str:
            input_str = self.input_str
        if not output_str:
            output_str = self.output_str

        input_features = ()
        for cc in self.input_str:
            if cc == 't':
                input_features += (trace_sample.inter_pkt_times_v,)
            elif cc == 'b':
                input_features += (trace_sample.inter_pkt_times_v * trace_sample.capacity_v,)
            elif cc == 's':
                input_features += (trace_sample.pkt_size_v,)
            elif cc == 'c':
                input_features += (trace_sample.capacity_v,)
            elif cc == 'q':
                input_features += (trace_sample.queue_bytes_v,)
            elif cc == 'l':
                print(f"WARNING: input feature: {cc} is not yet implemented")
            else:
                print(f"WARNING: input feature: {cc} is not recognized")

        output_features = ()
        for cc in self.output_str:
            if cc == 'b':
                output_features += (trace_sample.backlog_v,)
            elif cc == 'l':
                output_features += (trace_sample.latency_v,)
            elif cc == 'd':
                output_features += (trace_sample.dropped_status,)
        return np.stack(input_features, axis=-1), np.stack(output_features, axis=-1)


    def get_sample(self, test_index=0, data_set_name='test'):
        return self.trace_data[data_set_name][test_index]

    def get_feature_vector(self, test_index=0, data_set_name='test'):
        return self.feature_vector_from_sample(self.trace_data[data_set_name][test_index])


    def generate_trace_data_set(self, num_samples, seq_length, data_set_name='train'):
        """
        Possible input features:
        - t = inter-packet times
        - b = inter-pkt time * capacity
        - s = packet size
        - c = capacity
        - q = queue (bytes)
        - l = baseline latency (not implemented yet)

        possible output features:
        - b = backlog
        - l = latency
        - d = drop
        :param num_samples:
        :param seq_length:
        :return:
        """
        dataX= []  # Input sequences
        dataY = []  # Target output values
        if data_set_name not in self.trace_data:
            self.trace_data[data_set_name] = []

        for _ in range(num_samples):
            trace_sample = self.generate_trace_sample(seq_length)
            input_features, output_features = self.feature_vector_from_sample(trace_sample)
            dataX.append(input_features)
            dataY.append(output_features)
            self.trace_data[data_set_name].append(trace_sample)

        dataX_tensor_v = torch.tensor(np.array(dataX), dtype=torch.float32)
        dataY_tensor_v = torch.tensor(np.array(dataY), dtype=torch.float32)
        print(f"Data shape: X - {dataX_tensor_v.shape}, Y - {dataY_tensor_v.shape}")
        return dataX_tensor_v, dataY_tensor_v

    def get_loader(self, data_set_name='train', seq_length=None):
        if seq_length not in self.loaders[data_set_name]:
            seq_length = min(self.loaders[data_set_name].keys())
        return self.loaders[data_set_name][seq_length]

    def create_multiloaders(self, num_training_samples, seq_lengths_training, num_val_samples, seq_lengths_val, num_test_samples, seq_lengths_test, batch_size=64, seed=None):
        self.seed = seed
        rng_backup = np.random.get_state()
        if self.seed:
            np.random.seed(self.seed)

        # Generate test data first, so it will always be the same, even
        # if we change the size of training or val sets.
        self.num_test_samples = num_test_samples
        self.seq_length_test = seq_lengths_test  # Length of each sequence
        for seq_length in seq_lengths_test:
            dataX_test_tensor_v, dataY_test_tensor_v = self.generate_trace_data_set(num_test_samples, seq_length, data_set_name='test')
            print(dataX_test_tensor_v.shape, dataY_test_tensor_v.shape)
            self.loaders['test'][seq_length] = data.DataLoader(TensorDataset(dataX_test_tensor_v, dataY_test_tensor_v),
                                                               shuffle=False, batch_size=batch_size)

        self.num_training_samples = num_training_samples
        self.seq_length_training = seq_lengths_training
        for seq_length in seq_lengths_training:
            dataX_tensor_v, dataY_tensor_v = self.generate_trace_data_set(num_training_samples, seq_length, data_set_name='train')
            print(dataX_tensor_v.shape, dataY_tensor_v.shape)
            self.loaders['train'][seq_length] = data.DataLoader(TensorDataset(dataX_test_tensor_v, dataY_test_tensor_v),
                                                                shuffle=False, batch_size=batch_size)

        self.num_val_samples = num_val_samples  # Number of validation samples
        self.seq_length_val = seq_lengths_val  # Length of each sequence
        for seq_length in seq_lengths_val:
            dataX_val_tensor_v, dataY_val_tensor_v = self.generate_trace_data_set(num_val_samples, seq_length, data_set_name='val')
            print(dataX_val_tensor_v.shape, dataY_val_tensor_v.shape)
            self.loaders['val'][seq_length] = data.DataLoader(TensorDataset(dataX_test_tensor_v, dataY_test_tensor_v),
                                                              shuffle=False, batch_size=batch_size)
        if self.seed:
            np.random.set_state(rng_backup)


    def create_loaders(self, num_training_samples, seq_length_training, num_val_samples, seq_length_val, num_test_samples, seq_length_test, batch_size=64, seed=None):
        self.seed = seed
        rng_backup = np.random.get_state()
        if self.seed:
            np.random.seed(self.seed)

        # Generate test data first, so it will always be the same, even
        # if we change the size of training or val sets.
        self.num_test_samples = num_test_samples
        self.seq_length_test = seq_length_test  # Length of each sequence
        dataX_test_tensor_v, dataY_test_tensor_v = self.generate_trace_data_set(num_test_samples, seq_length_test, data_set_name='test')
        print(dataX_test_tensor_v.shape, dataY_test_tensor_v.shape)

        self.num_training_samples = num_training_samples
        self.seq_length_training = seq_length_training
        dataX_tensor_v, dataY_tensor_v = self.generate_trace_data_set(num_training_samples, seq_length_training, data_set_name='train')
        print(dataX_tensor_v.shape, dataY_tensor_v.shape)

        self.num_val_samples = num_val_samples  # Number of validation samples
        self.seq_length_val = seq_length_val  # Length of each sequence
        dataX_val_tensor_v, dataY_val_tensor_v = self.generate_trace_data_set(num_val_samples, seq_length_val, data_set_name='val')
        print(dataX_val_tensor_v.shape, dataY_val_tensor_v.shape)

        if self.seed:
            np.random.set_state(rng_backup)

        self.loaders['train'][seq_length_training] = data.DataLoader(data.TensorDataset(dataX_tensor_v, dataY_tensor_v),
                                                                       shuffle=True, batch_size=batch_size)
        self.loaders['val'][seq_length_val] = data.DataLoader(data.TensorDataset(dataX_val_tensor_v, dataY_val_tensor_v),
                                                                shuffle=False, batch_size=batch_size)
        self.loaders['test'][seq_length_test] = data.DataLoader(TensorDataset(dataX_test_tensor_v, dataY_test_tensor_v),
                                                                  shuffle=False, batch_size=batch_size)

    def plot_inputs(self, index=0, data_set_name='train'):
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
        arrival_times_v = self.trace_data[data_set_name][index].pkt_arrival_times_v
        backlog_v = self.trace_data[data_set_name][index].backlog_v
        pkt_size_v = self.trace_data[data_set_name][index].pkt_size_v
        dropped_status = self.trace_data[data_set_name][index].dropped_status
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



