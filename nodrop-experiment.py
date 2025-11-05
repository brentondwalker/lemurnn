#!/usr/bin/env python3

import argparse

from BacklogLSTM import BacklogLSTM
from BacklogRNN import BacklogRNN
from BacklogReluLSTM import BacklogReluLSTM
from DropLSTM import DropLSTM
from DropReluLSTM import DropReluLSTM
from LatencyPredictorNodrop import LatencyPredictorNodrop
from LinkEmuModel import LinkEmuModel
from LinkProperties import link_properties_library
from NonManualRNN import NonManualRNN
from TraceGenerator import *
from TraceGeneratorCodel import TraceGeneratorCodel
from LatencyPredictor import *
from LatencyPredictorEarthmover import LatencyPredictorEarthmover
from LatencyPredictorEnergy import LatencyPredictorEnergy


def main():
    # configure:
    #num layers
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_properties', type=str, default='default')
    #parser.add_argument('--infinite_queue', action='store_true')
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
    parser.add_argument("-d", '--dropout_rate', type=float, default=0.0)
    parser.add_argument("-a", '--compute_ads_loss', action='store_true')
    #parser.add_argument('--codel', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--energy', action='store_true')
    parser.add_argument('--earthmover', action='store_true')
    parser.add_argument('--tanh', action='store_true')
    parser.add_argument('--relu_lstm', action='store_true')
    parser.add_argument('--lstm', action='store_true')

    args = parser.parse_args()

    link_properties_str = args.link_properties
    infinite_queue = True
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
    dropout_rate = args.dropout_rate
    data_seed = args.data_seed
    torch_seed = args.torch_seed
    #codel = args.codel
    energy = args.energy
    earthmover = args.earthmover
    use_relu_lstm = args.relu_lstm
    use_lstm = args.lstm
    normalize = args.normalize
    nonlinearity = 'relu'
    if args.tanh:
        nonlinearity = 'tanh'
    if compute_ads_loss:
        ads_loss_interval = 100

    link_properties = link_properties_library[link_properties_str]
    if infinite_queue:
        link_properties.infinite_queue()

    #trace_generator = TraceGenerator(link_properties, input_str='bs', output_str='b', normalize=normalize)
    trace_generator = TraceGenerator(link_properties, normalize=normalize)

    #trace_generator.create_multiloaders(1024*kilo_training_samples, [4, 8, 16, 32, 64, 128, 256],
    #                               1024*kilo_val_samples, [seq_len],
    #                               1024*kilo_test_samples, [seq_len],   #1024*kilo_test_samples, seq_len*2,
    #                               seed=data_seed)
    trace_generator.create_multiloaders(1024 * kilo_training_samples, [4, 8, 16, 32, 64, 128, 256],
                                   1024*kilo_val_samples, [seq_len],
                                   1024*kilo_test_samples, [seq_len],   #1024*kilo_test_samples, seq_len*2,
                                   seed=data_seed)

    if use_relu_lstm:
        model:LinkEmuModel = BacklogReluLSTM(input_size=trace_generator.input_size(),
                                          hidden_size=hidden_size, num_layers=num_layers,
                                          learning_rate=learning_rate)
    elif use_lstm:
        model: LinkEmuModel = BacklogLSTM(input_size=trace_generator.input_size(),
                                           hidden_size=hidden_size, num_layers=num_layers,
                                           learning_rate=learning_rate, dropout_rate=dropout_rate)
    else:
        model:LinkEmuModel = BacklogRNN(input_size=trace_generator.input_size(),
                                          hidden_size=hidden_size, num_layers=num_layers,
                                          learning_rate=learning_rate, dropout_rate=dropout_rate,
                                          nonlinearity=nonlinearity)
    model.set_optimizer()

    latency_predictor = LatencyPredictorNodrop(model, trace_generator=trace_generator, seed=torch_seed)

    latency_predictor.train(learning_rate=learning_rate, n_epochs=num_epochs, ads_loss_interval=ads_loss_interval)



# ======================================
# ======================================
# ======================================

if __name__ == "__main__":
    main()
