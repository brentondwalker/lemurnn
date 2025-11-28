# lemurnn
**L**ink **EM**ulation **U**sing **RNN**s

Based on the Jupyter Notebooks of Shuwen Yang

# Set up the virtual environment

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```


# Create and train a model (synthetic data)

```
usage: rnn-loss-experiment.py [-h] [--link_properties LINK_PROPERTIES] [--infinite_queue] [-l NUM_LAYERS]
                              [-s HIDDEN_SIZE] [-e NUM_EPOCHS] [--kilo_training_samples KILO_TRAINING_SAMPLES]
                              [--kilo_val_samples KILO_VAL_SAMPLES] [--kilo_test_samples KILO_TEST_SAMPLES]
                              [--seq_len SEQ_LEN] [--data_seed DATA_SEED] [--torch_seed TORCH_SEED] [-r LEARNING_RATE]
                              [-d DROPOUT_RATE] [-a] [--dag_data DAG_DATA] [--codel] [--packetqueue] [--energy]
                              [--earthmover] [--tanh] [--relu_lstm] [--lstm] [--normalize] [--multiloader]
                              [--drop_masking]

options:
  -h, --help            show this help message and exit
  --link_properties LINK_PROPERTIES
  --infinite_queue
  -l NUM_LAYERS, --num_layers NUM_LAYERS
  -s HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
  --kilo_training_samples KILO_TRAINING_SAMPLES
  --kilo_val_samples KILO_VAL_SAMPLES
  --kilo_test_samples KILO_TEST_SAMPLES
  --seq_len SEQ_LEN
  --data_seed DATA_SEED
  --torch_seed TORCH_SEED
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
  -d DROPOUT_RATE, --dropout_rate DROPOUT_RATE
  -a, --compute_ads_loss
  --dag_data DAG_DATA
  --codel
  --packetqueue
  --energy
  --earthmover
  --tanh
  --relu_lstm
  --lstm
  --normalize
  --multiloader
  --drop_masking
```
## Examples

The default will train:
- RNN with ReLU activation
- layers=1, hidden_size=8
- Inputs: bscq
- Outputs: bd
- Synthetic bytequeue data
- Default link properties: arrival_rate=U(0.2, 1), capacity=U(500, 1000), pkt_size=U(500, 1500), queue_bytes=U(2500,10000)
- num_epochs=100
- seq_length=128
- Loss function = L1(backlog) + CrossEntropy(drops)
- Data size: training=2048, validation=1024, test=1024
- No normalizaiton

`./rnn-loss-experiment.py`

More extensive example
```
/rnn-loss-experiment.py -e 2000 -a -l 3 -s 64 --kilo_training_samples 8 --kilo_val_samples 2 --kilo_test_samples 2 --seq_len 128 -r 0.001 --data_seed 10 --torch_seed $SEED --codel --multiloader --earthmover --lstm --normalize
```

## Output data
The training script is very prolific in what it saves.
- It will try to create a subdirectory under **model-training/**
- The output directory will be named: **model-<trainer_info>-<model_name>-l<layers>_h<hidden_size>-<datatype>_<inputs>_<outputs>_<loader_info>-<timestamp>**
- For example: model-trainer_bx-droprelurnn-l1_h8-bytequeue_bscq_bd_multiloader-1764360236/
- At the start of every run it will save:
  - **model-properties.json**
  - **dataset-properties.json**
  - **link-properties.json**
- Every epoch it saves:
  - a line of json to **training_history.json** with all current loss functions and other metrics for train, val, and test sets.
  - a line of the loss and metric data in CSV to **training_log.dat**
- Every epoch in which it finds a new best model:
  - **modelstate-<epoch>.json** the model state dict (not actually json)
  - **modelstate-torchscript-<epoch>.pt** model traced and exported by torchscript.  Can be loaded by libtorch.
  - **BD_plot_test_sample0_epoch<epoch>.png** - a plot of the current model prediction and ground truth for sample 0 of the test set.
- If gradient tracking is enabled (track_grad), then it dumps a huge ammt of gradient data from the training loop to **grad-tracker-{self.name}.dat**
  - Not implemented on the command line.  To get this you have to enable it in the source code.  but if you are digging this deep, this should be no problem.

# Trace data
You can train and experiment using samples loaded from trace files.

The trace files have the format:
- packet_number
- size [Bytes]
- transmit_time [epoch seconds, float]
- receive_time [epoch seconds, float, 0 if dropped]
- latency [seconds, float]
- dropped_status [1 or 0]

For example:
```
packet_number   size    transmit_time   receive_time    latency dropped_status
1       1396    1764001042.4637475      1764001042.4650688      0.0013213157653808594   0
2       1358    1764001042.7929063      1764001042.7941916      0.0012853145599365234   0
3       587     1764001042.8111677      1764001042.8117487      0.0005810260772705078   0
4       1123    1764001042.8986487      1764001042.899719       0.001070261001586914    0
[...]
```

You get them into the trainer with the `--dag_data` option.  The script will glob all *.csv files under the path passed in with that option (including symlinks) and use them to generate train, val, and test sets.

Example:
```
./rnn-loss-experiment.py -e 2000  -l 4 -s 64 --kilo_training_samples 8 --kilo_val_samples 2 --kilo_test_samples 2 
--seq_len 1024 -r 0.001 --data_seed 10 --torch_seed 111 --multiloader --earthmover --drop_masking  --dag_data dag-traces/
bytequeue-traces/ -d 0.2
```

Important notes:
- The loader will randomly permute the list of files before it loads them.
- If there are not enough files for the number samples you requested, it will re-premute the files and load them again.
- It will only load the requested seq_length.  If the file has data for 4096 packets, but you ask for 128, you will always get the first 128.
- This is intended to be used with a multiloader, which loads sequences of different lengths.  In this case, several different-length prefixes of each sequence will end up getting loaded.
- TODO: it does not separate the initial set of trace files into train, val, and test.  So if it recycles samples, your val and test sets will get poluted.

# Run a model against a trace file

Model must be in a torchscript-encoded *.pt file.
You must tell the program the number of layers, hidden size, and whether or not the model is an LSTM.  The program cannot deduce these from the model file.

```
usage: run-model.py [-h] [-m MODEL] [-l NUM_LAYERS] [-s HIDDEN_SIZE] [--seq_len SEQ_LEN] [--dag_data DAG_DATA]
                    [--packetqueue] [--lstm] [--normalize]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
  -l NUM_LAYERS, --num_layers NUM_LAYERS
  -s HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
  --seq_len SEQ_LEN
  --dag_data DAG_DATA
  --packetqueue
  --lstm
  --normalize
```

Example:
```
run-model.py -m example_models/modelstate-torchscript-1632.pt -l 4 -s 64 --lstm --normalize --seq_len 4096 --dag_data mgtrace_C7_L0_Q69350_1763732340_17.csv
```

# Run the forwarder

## Building
Install dependencies (certainly forgetting some):
```
sudo apt update
sudo apt install build-essential cmake dpdk-dev 
```

Get libtorch:
```
cd forwarder
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

Build the software:
```
cd build
cmake -DCMAKE_PREFIX_PATH=./libtorch ..
make
```

## Run the forwarder

First take down the interfaces you intend to use for DPDK.  Or just take down all unused interfaces, and bind them to a DPDK driver.  Also allocate hugepages.
```
sudo ifconfig <interface_1> down
[...]
sudo ifconfig <interface_n> down
sudo bind-interfaces.sh
sudo setup-hugetlbfs.sh
dpdk-devbind.py --status
```

Finally, run it.
```
Usage: [DPDK ARGS] -- <dev1> <dev2> -h <hidden_size> -l <layers> -m <model_file> -c <capacity(Mb/s)> -q <queuesize(KB)> -t <model_type>
```
Example:
```
sudo ./lemu_forwarder -- 1 2 -h 64 -l 4 -m ../example_models/modelstate-torchscript-1632.pt -c 2 -q 5 -t lstm
```

## Look at the forwarder behavior

If you give the forwarder the `-f <filename>` argument, it will save off info about each packet it sees, and how it was handled.  The format of the file is:
1. packet_idx
1. inter_packet_time_ms
1. processed_kbit
1. size_byte
1. pa.latency_ms
1. pa.drop
1. prediction_time_ms
1. num_drops
1. arrival_tsc
1. arrival_ms
1. inter_packet_time_tsc
1. size_kbyte
1. send_time_tsc

