#!/usr/bin/env python3

import argparse
from TraceGenerator import *
from LatencyPredictor import *



def main():
    # configure:
    #num layers
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", '--num_layers', type=int, default=1)
    parser.add_argument("-s", '--hidden_size', type=int, default=8)
    parser.add_argument("-e", '--num_epochs', type=int, default=100)
    parser.add_argument('--kilo_training_samples', type=int, default=2)
    parser.add_argument('--kilo_val_samples', type=int, default=1)
    parser.add_argument('--kilo_test_samples', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--data_seed', type=int, default=None)
    parser.add_argument('--torch_seed', type=int, default=None)
    parser.add_argument("-r", '--learning_rate', type=float, default=0.001)
    parser.add_argument("-a", '--compute_ads_loss', action='store_true')

    args = parser.parse_args()

    num_layers = args.num_layers
    hidden_size = args.hidden_size
    num_epochs = args.num_epochs
    compute_ads_loss = args.compute_ads_loss
    ads_loss_interval = 0
    kilo_training_samples = args.kilo_training_samples
    kilo_val_samples = args.kilo_val_samples
    kilo_test_samples = args.kilo_test_samples
    seq_len = args.seq_len
    learning_rate = args.learning_rate
    data_seed = args.data_seed
    torch_seed = args.torch_seed
    if compute_ads_loss:
        ads_loss_interval = 100

    link_properties = LinkProperties(min_arrival_rate=0.2,
                        max_arrival_rate=1,
                        min_capacity=500,
                        max_capacity=1000,
                        min_pkt_size=500,
                        max_pkt_size=1500,
                        min_queue_bytes=2500,
                        max_queue_bytes=10000)

    trace_generator = TraceGenerator(link_properties)

    trace_generator.create_loaders(1024*kilo_training_samples, seq_len,
                                   1024*kilo_val_samples, seq_len,
                                   1024*kilo_test_samples, seq_len*2,
                                   seed=data_seed)

    latency_predictor = LatencyPredictor(hidden_size=hidden_size, num_layers=num_layers, trace_generator=trace_generator, seed=torch_seed)

    latency_predictor.train(learning_rate=learning_rate, n_epochs=num_epochs, ads_loss_interval=ads_loss_interval)



# ======================================
# ======================================
# ======================================

if __name__ == "__main__":
    main()
