#!/usr/bin/env python3

import argparse
import sys
import wandb

from DropGRU import DropGRU
from DropGRUAR import DropGRUAR
from DropLSTM import DropLSTM
from DropLSTMAR import DropLSTMAR
from DropReluLSTM import DropReluLSTM
from LatencyPredictorEarthmoverAR import LatencyPredictorEarthmoverAR
from LatencyPredictorTTS import LatencyPredictorTTS
from LatencyPredictorTTSAR import LatencyPredictorTTSAR
from LinkEmuModel import LinkEmuModel
from LinkProperties import link_properties_library
from NonManualRNN import NonManualRNN
from NonManualRNNAR import NonManualRNNAR
from TraceGeneratorByteQueue import TraceGeneratorByteQueue
from TraceGeneratorCodel import TraceGeneratorCodel
from LatencyPredictor import *
from LatencyPredictorEarthmover import LatencyPredictorEarthmover
from LatencyPredictorEnergy import LatencyPredictorEnergy
from TraceGeneratorDagData import TraceGeneratorDagData
from TraceGeneratorPacketQueue import TraceGeneratorPacketQueue


def setup_wandb(wandb_cfg:dict):
    # get the current run counter so we can set a custom name
    api = wandb.Api()
    runs = api.runs("brenton-d-walker-no/lemurnn")
    suffix = ""
    if wandb_cfg['wandb_suffix'] is not None:
        suffix = f"-{wandb_cfg['wandb_suffix']}"
    run_name = f"{wandb_cfg['model']}-{wandb_cfg['num_layers']}-{wandb_cfg['hidden_size']}-{len(wandb_cfg['traffic_types'])}{suffix}-{len(runs)}"
    print(f"Using wandb run name: \"{run_name}\"")
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="brenton-d-walker-no",
        # Set the wandb project where this run will be logged.
        project="lemurnn",
        # Track hyperparameters and run metadata.
        config=wandb_cfg,
        # include this too for some reason
        job_type="training",
        # by default wandb give the runs random names that are confusing when looking at results
        name=run_name
    )
    return run

def main():
    # configure:
    #num layers
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_properties', type=str, action='append', default=None)
    parser.add_argument('--traffic', type=str, action='append', default=None)
    parser.add_argument('--infinite_queue', action='store_true')
    parser.add_argument("-l", '--num_layers', type=int, default=1)
    parser.add_argument("-s", '--hidden_size', type=int, default=8)
    parser.add_argument("-e", '--num_epochs', type=int, default=100)
    parser.add_argument('--kilo_training_samples', type=int, default=2)
    parser.add_argument('--kilo_val_samples', type=int, default=1)
    parser.add_argument('--kilo_test_samples', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--tb_chunk_size', type=int, default=None)
    parser.add_argument('--data_seed', type=int, default=None)
    parser.add_argument('--torch_seed', type=int, default=None)
    parser.add_argument("-r", '--learning_rate', type=float, default=0.001)
    parser.add_argument("-d", '--dropout_rate', type=float, default=0.0)
    parser.add_argument("-a", '--compute_ads_loss', action='store_true')
    parser.add_argument('--dag_data', type=str, action='append', default=None)
    parser.add_argument('--codel', action='store_true')
    parser.add_argument('--packetqueue', action='store_true')
    parser.add_argument('--energy', action='store_true')
    parser.add_argument('--earthmover', action='store_true')
    parser.add_argument('--tts', action='store_true')
    parser.add_argument('--tanh', action='store_true')
    parser.add_argument('--relu_lstm', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--multiloader', action='store_true')
    parser.add_argument('--drop_masking', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_suffix', type=str, default=None)
    parser.add_argument('--autoregressive', action='store_true')
    parser.add_argument('--use_deltas', action='store_true')

    args = parser.parse_args()

    link_properties_strs = args.link_properties
    infinite_queue = args.infinite_queue
    traffic_types = args.traffic
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    num_epochs = args.num_epochs
    compute_ads_loss = args.compute_ads_loss
    ads_loss_interval = 0
    kilo_training_samples = args.kilo_training_samples
    kilo_val_samples = args.kilo_val_samples
    kilo_test_samples = args.kilo_test_samples
    seq_len = args.seq_len
    tb_chunk_size = args.tb_chunk_size
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    data_seed = args.data_seed
    torch_seed = args.torch_seed
    codel = args.codel
    packetqueue = args.packetqueue
    energy = args.energy
    earthmover = args.earthmover
    temporal_target_smearing = args.tts
    use_relu_lstm = args.relu_lstm
    use_lstm = args.lstm
    use_gru = args.gru
    normalize = args.normalize
    multiloader = args.multiloader
    drop_masking = args.drop_masking
    dag_data = args.dag_data
    nonlinearity = 'relu'
    if args.tanh:
        nonlinearity = 'tanh'
    if compute_ads_loss:
        ads_loss_interval = 100
    use_wandb = args.wandb
    wandb_suffix = args.wandb_suffix
    autoregressive = args.autoregressive
    use_deltas = args.use_deltas

    if link_properties_strs is None:
        link_properties_strs = ['default']
    link_properties = [link_properties_library[lps] for lps in link_properties_strs]
    if infinite_queue:
        for lp in link_properties:
            lp.infinite_queue()
    #if traffic_types is None:
    #    print("Using default traffic type of exponential")
    #    traffic_types = ['exponential']

    trace_generator = None
    if dag_data:
        # assign fixed value to packet size, because that will be used to rescale the predictions
        link_properties[0].max_pkt_size = 1000
        link_properties[0].min_pkt_size = 1000
        trace_generator = TraceGeneratorDagData(link_properties, normalize=normalize, datadirs=dag_data)
    else:
        if packetqueue:
            trace_generator = TraceGeneratorPacketQueue(link_properties, normalize=normalize, traffic_types=traffic_types)
        elif codel:
            trace_generator = TraceGeneratorCodel(link_properties, normalize=normalize, base_interval=10, codel_threshold=5)
        else:
            # default is ByteQueue
            # trace_generator = TraceGenerator(link_properties, normalize=normalize)
            trace_generator = TraceGeneratorByteQueue(link_properties, normalize=normalize, traffic_types=traffic_types)

    if multiloader:
        seq_lengths_training = []
        sl = 4
        while sl <= seq_len:
            seq_lengths_training.append(sl)
            sl *= 2
        trace_generator.create_multiloaders(1024*kilo_training_samples, seq_lengths_training,
                                       1024*kilo_val_samples, [seq_len],
                                       1024*kilo_test_samples, [seq_len],   #1024*kilo_test_samples, seq_len*2,
                                       seed=data_seed)
    else:
        trace_generator.create_multiloaders(1024 * kilo_training_samples, [seq_len],
                                       1024*kilo_val_samples, [seq_len],
                                       1024*kilo_test_samples, [seq_len],   #1024*kilo_test_samples, seq_len*2,
                                       seed=data_seed)

    if use_relu_lstm:
        model:LinkEmuModel = DropReluLSTM(input_size=trace_generator.input_size(),
                                          hidden_size=hidden_size, num_layers=num_layers,
                                          learning_rate=learning_rate, dropout_rate=dropout_rate)
    elif use_lstm:
        print("USING LSTM!!")
        if autoregressive:
            model: LinkEmuModel = DropLSTMAR(input_size=trace_generator.input_size(),
                                             hidden_size=hidden_size, num_layers=num_layers,
                                             learning_rate=learning_rate, dropout_rate=dropout_rate, use_deltas=use_deltas)
        else:
            model: LinkEmuModel = DropLSTM(input_size=trace_generator.input_size(),
                                           hidden_size=hidden_size, num_layers=num_layers,
                                           learning_rate=learning_rate, dropout_rate=dropout_rate)
    elif use_gru:
        print("USING GRU!!")
        if autoregressive:
            model: LinkEmuModel = DropGRUAR(input_size=trace_generator.input_size(),
                                            hidden_size=hidden_size, num_layers=num_layers,
                                            learning_rate=learning_rate, dropout_rate=dropout_rate, use_deltas=use_deltas)
        else:
            model: LinkEmuModel = DropGRU(input_size=trace_generator.input_size(),
                                          hidden_size=hidden_size, num_layers=num_layers,
                                          learning_rate=learning_rate, dropout_rate=dropout_rate)
    else:
        if autoregressive:
            model:LinkEmuModel = NonManualRNNAR(input_size=trace_generator.input_size(),
                                              hidden_size=hidden_size, num_layers=num_layers,
                                              learning_rate=learning_rate, dropout_rate=dropout_rate,
                                              nonlinearity=nonlinearity, use_deltas=use_deltas)
        else:
            model: LinkEmuModel = NonManualRNN(input_size=trace_generator.input_size(),
                                               hidden_size=hidden_size, num_layers=num_layers,
                                               learning_rate=learning_rate, dropout_rate=dropout_rate,
                                               nonlinearity=nonlinearity)
    model.set_optimizer()
    print(f"MODEL NAME IS: {model.get_model_name()}")

    wandb_run = None
    if use_wandb:
        wandb_cfg = {
            'model':            model.get_model_name(),
            'num_layers':       num_layers,
            'hidden_size':      hidden_size,
            'learning_rate':    learning_rate,
            'nonlinearity':     nonlinearity,
            'dropout_rate':     dropout_rate,
            'drop_masking':     drop_masking,
            'link_properties':  link_properties_strs,
            'traffic_types':    traffic_types,
            'trace_generator':  trace_generator.data_type,
            'kilo_training_samples': kilo_training_samples,
            'kilo_val_samples': kilo_val_samples,
            'kilo_test_samples':    kilo_test_samples,
            'multiloader':      multiloader,
            'autoregressive':   autoregressive,
            'earthmover':       earthmover,
            'temporal_target_smearing': temporal_target_smearing,
            'use_deltas':       use_deltas,
            'seq_len':          seq_len,
            'tb_chunk_size':    tb_chunk_size,
            'data_seed':        data_seed,
            'torch_seed':       torch_seed,
            'training_directory': model.training_directory,
            'wandb_suffix':     wandb_suffix
        }
        wandb_run = setup_wandb(wandb_cfg)
        wandb.watch(model)

    if temporal_target_smearing:
        if autoregressive:
            latency_predictor = LatencyPredictorTTSAR(model, trace_generator=trace_generator, seed=torch_seed,
                                                      drop_masking=drop_masking, wandb_run=wandb_run, use_deltas=use_deltas, tb_chunk_size=tb_chunk_size)
        else:
            latency_predictor = LatencyPredictorTTS(model, trace_generator=trace_generator, seed=torch_seed,
                                                    drop_masking=drop_masking, wandb_run=wandb_run, tb_chunk_size=tb_chunk_size)
    elif earthmover:
        if autoregressive:
            latency_predictor = LatencyPredictorEarthmoverAR(model, trace_generator=trace_generator, seed=torch_seed, drop_masking=drop_masking, wandb_run=wandb_run, use_deltas=use_deltas, tb_chunk_size=tb_chunk_size)
        else:
            latency_predictor = LatencyPredictorEarthmover(model, trace_generator=trace_generator, seed=torch_seed, drop_masking=drop_masking, wandb_run=wandb_run, tb_chunk_size=tb_chunk_size)
    elif energy:
        latency_predictor = LatencyPredictorEnergy(model, trace_generator=trace_generator, seed=torch_seed)
    else:
        latency_predictor = LatencyPredictor(model, trace_generator=trace_generator, seed=torch_seed)

    latency_predictor.train(n_epochs=num_epochs, ads_loss_interval=ads_loss_interval)

    if wandb_run:
        wandb_run.finish()

# ======================================
# ======================================
# ======================================

if __name__ == "__main__":
    main()
