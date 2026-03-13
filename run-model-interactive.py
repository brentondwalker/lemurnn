#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from LinkProperties import link_properties_library
from LatencyPredictor import *
from TraceGeneratorByteQueue import TraceGeneratorByteQueue
from TraceGeneratorCodel import TraceGeneratorCodel
from TraceGeneratorDagData import TraceGeneratorDagData
from TraceGenerator import TraceGenerator
from TraceGeneratorPacketQueue import TraceGeneratorPacketQueue
import pytorch_stats_loss as stats_loss


class InteractiveEvaluator:
    def __init__(self, model, trace_generator, link_properties, traffic_types, seq_len, device, normalize,
                 normalize_earthmover):
        self.model = model
        self.trace_generator = trace_generator
        self.link_properties = link_properties
        self.traffic_types = traffic_types
        self.seq_len = seq_len
        self.device = device
        self.normalize = normalize
        self.normalize_earthmover = normalize_earthmover

        # State tracking for interactivity
        self.history = []
        self.current_index = 0

        # Set up the Matplotlib figure and axis
        plt.ioff()  # Turn off interactive mode so it doesn't draw prematurely
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        """Handle right and left arrow key presses."""
        if event.key == 'right':
            self.current_index += 1
            self.update_plot()
        elif event.key == 'left':
            if self.current_index > 0:
                self.current_index -= 1
                self.update_plot()
            else:
                print("\n[Already at the first sample. Press right arrow to generate more.]")

    def update_plot(self):
        """Runs inference on the current sample (or generates a new one) and updates the plot."""
        # Fetch from history or generate a new trace sample
        if self.current_index >= len(self.history):
            print(f"\n--- Generating New Sample (Index {self.current_index}) ---")
            trace_sample = self.trace_generator.generate_trace_sample(self.link_properties, self.traffic_types[0],
                                                                      self.seq_len)
            self.history.append(trace_sample)
        else:
            print(f"\n--- Loading Sample from History (Index {self.current_index}) ---")
            trace_sample = self.history[self.current_index]

        # Data Preparation
        input_v, output_v = self.trace_generator.feature_vector_from_sample(trace_sample)

        batch_size = 1
        hidden = self.model.new_hidden_tensor(batch_size, self.device)
        dataX = torch.tensor(input_v, dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        dataY = torch.tensor(output_v, dtype=torch.float32).unsqueeze(dim=0).to(self.device)

        self.trace_generator.print_means(dataX.cpu(), dataY.cpu())
        print(f"dataX: {dataX.shape}\tdataY: {dataY.shape}")

        # Inference
        backlog_pred, dropped_pred, _ = self.model(dataX, hidden)
        print(f"backlog_pred: {backlog_pred.shape}\tdropped_pred: {dropped_pred.shape}")

        # Reshape outputs
        dropped_pred_binary = torch.argmax(dropped_pred, dim=2)
        true_backlog = dataY[:, :, 0].squeeze().cpu().numpy()
        true_drops = dataY[:, :, 1].squeeze().cpu().numpy()
        predicted_backlog = backlog_pred.squeeze().cpu().detach().numpy()
        predicted_drops = dropped_pred_binary.squeeze().cpu().detach().numpy()

        if self.normalize:
            true_backlog *= 1000
            predicted_backlog *= 1000

        # Update the Plot
        self.ax.clear()  # Clear the previous plot elements
        pkt_arrival_times_v = trace_sample.pkt_arrival_times_v[0:self.seq_len]

        self.ax.plot(pkt_arrival_times_v, true_backlog, label="Generated Backlog", color='green', linewidth=2.5,
                     zorder=1)
        self.ax.plot(pkt_arrival_times_v, predicted_backlog, label="Predicted Backlog", linestyle="dashed", color='red',
                     linewidth=2.5, zorder=1)

        # Real dropped packet positions
        drop_indices_real = np.where(true_drops == 1)[0]
        self.ax.scatter(pkt_arrival_times_v[drop_indices_real], true_backlog[drop_indices_real], color='blue',
                        marker='x', label="Real Dropped Packets", linewidth=2.5, zorder=2)

        # Predicted dropped packet positions
        drop_indices_pred = np.where(predicted_drops == 1)[0]
        self.ax.scatter(pkt_arrival_times_v[drop_indices_pred], predicted_backlog[drop_indices_pred], color='orange',
                        marker='o', label="Predicted Dropped Packets", linewidth=2, zorder=2)

        self.ax.set_xlabel("Time [ms]", fontsize=18)
        self.ax.set_ylabel("Backlog [KByte]", fontsize=18)
        self.ax.set_title(f"Sample {self.current_index + 1} of {len(self.history)}")
        self.ax.legend()
        self.ax.grid()
        self.fig.canvas.draw()  # Force Matplotlib to redraw the canvas

        # Print Stats
        pred_dropped_status = predicted_drops.astype(int)
        real_dropped_status = true_drops.astype(int)
        pred_indices = set(np.where(pred_dropped_status == 1)[0])
        real_indices = set(np.where(real_dropped_status == 1)[0])
        matched = pred_indices & real_indices
        correct = len(matched)
        total = len(real_indices)
        accuracy = correct / total * 100 if total > 0 else 0.0

        print(f"Drop position match accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Number of drops:  real={len(real_indices)}  predicted={len(pred_indices)}")

        try:
            tensor_w1 = torch.from_numpy(real_dropped_status.astype(dtype=np.float64))
            tensor_w2 = torch.from_numpy(pred_dropped_status.astype(dtype=np.float64))
            print("\n\nWasserstein Results:")
            emp1_loss = torch.sum(
                stats_loss.symmetric_earthmover(tensor_w1, tensor_w2, p=1, normalize=self.normalize_earthmover))
            emp2_loss = torch.sum(
                stats_loss.symmetric_earthmover(tensor_w1, tensor_w2, p=2, normalize=self.normalize_earthmover))
            print(f"Symmetric earthmover loss p1: {emp1_loss}\tp2: {emp2_loss}")
            print("Wasserstein loss", stats_loss.torch_wasserstein_loss(tensor_w1, tensor_w2).data)
            print("Energy loss", stats_loss.torch_energy_loss(tensor_w1, tensor_w2).data)
            print("p == 1.5 CDF loss", stats_loss.torch_cdf_loss(tensor_w1, tensor_w2, p=1.5).data)
            print("Validate Checking Errors:", stats_loss.torch_validate_distibution(tensor_w1, tensor_w2))
        except NameError:
            pass  # Fails gracefully if stats_loss is not imported properly

    def show(self):
        """Triggers the first plot and starts the interactive Matplotlib loop."""
        self.update_plot()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', type=str, default=None)
    parser.add_argument("-w", '--wandb_run', type=str, default=None)
    parser.add_argument("-l", '--num_layers', type=int, default=1)
    parser.add_argument("-s", '--hidden_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--dag_data', type=str, action='append', default=None)
    parser.add_argument('--data_seed', type=int, default=None)
    parser.add_argument('--link_properties', type=str, default='daglike')
    parser.add_argument('--traffic', type=str, action='append', default=None)
    parser.add_argument('--codel', action='store_true')
    parser.add_argument('--packetqueue', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--normalize_earthmover', action='store_true', default=False)
    parser.add_argument('--autoregressive', action='store_true')
    parser.add_argument('--use_deltas', action='store_true')

    args = parser.parse_args()

    # Configurations
    model_file = args.model
    wandb_run = args.wandb_run
    seq_len = args.seq_len
    normalize = args.normalize
    normalize_earthmover = args.normalize_earthmover
    dag_data = args.dag_data
    link_properties_str = args.link_properties
    data_seed = args.data_seed
    codel = args.codel
    packetqueue = args.packetqueue
    traffic_types = args.traffic if args.traffic else ['exponential']

    link_properties = link_properties_library[link_properties_str]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # Generator Setup
    if dag_data:
        link_properties[0].max_pkt_size = 1000
        link_properties[0].min_pkt_size = 1000
        trace_generator = TraceGeneratorDagData(link_properties, normalize=normalize, datadirs=dag_data)
    elif packetqueue:
        trace_generator = TraceGeneratorPacketQueue(link_properties, normalize=normalize, traffic_types=traffic_types)
    elif codel:
        trace_generator = TraceGeneratorCodel(link_properties, normalize=normalize, base_interval=10, codel_threshold=5)
    else:
        trace_generator = TraceGeneratorByteQueue(link_properties, normalize=normalize, traffic_types=traffic_types)

    if data_seed:
        np.random.seed(data_seed)

    # Load Model
    if wandb_run is not None:
        model = LinkEmuModel.load_model_wandb(wandb_run, device)
    elif model_file is not None:
        model = torch.jit.load(model_file)
    else:
        raise ValueError(f"Must specify either '-m <model_file>' or '-w wandb_run_identifier'")
    model.eval()

    # Launch Interactive Evaluator
    print("\n--- Starting Interactive Evaluation ---")
    print("Use the RIGHT ARROW key to generate/evaluate a new sample.")
    print("Use the LEFT ARROW key to view previous samples.")

    evaluator = InteractiveEvaluator(
        model=model,
        trace_generator=trace_generator,
        link_properties=link_properties,
        traffic_types=traffic_types,
        seq_len=seq_len,
        device=device,
        normalize=normalize,
        normalize_earthmover=normalize_earthmover
    )

    evaluator.show()


if __name__ == "__main__":
    main()