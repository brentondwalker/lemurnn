#!/usr/bin/env python3

import argparse
from LinkProperties import link_properties_library
from LatencyPredictor import *
from TraceGeneratorByteQueue import TraceGeneratorByteQueue
from TraceGeneratorCodel import TraceGeneratorCodel
from TraceGeneratorDagData import TraceGeneratorDagData
from TraceGenerator import TraceGenerator
from TraceGeneratorPacketQueue import TraceGeneratorPacketQueue

"""
Run a model on a single trace file and plot the prediction vs the truth.
"""

def main():
    # configure:
    #num layers
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', type=str, default=None)
    parser.add_argument("-l", '--num_layers', type=int, default=1)
    parser.add_argument("-s", '--hidden_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--dag_data', type=str, action='append', default=None)
    parser.add_argument('--data_seed', type=int, default=None)
    parser.add_argument('--link_properties', type=str, default='default')
    parser.add_argument('--codel', action='store_true')
    parser.add_argument('--packetqueue', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--normalize', action='store_true')

    args = parser.parse_args()

    model_file = args.model
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    seq_len = args.seq_len
    use_lstm = args.lstm
    normalize = args.normalize
    sample_data = args.dag_data
    link_properties_str = args.link_properties
    data_seed = args.data_seed
    codel = args.codel
    packetqueue = args.packetqueue

    link_properties = link_properties_library[link_properties_str]

    # load the data sample
    if sample_data:
        trace_generator = TraceGeneratorDagData(link_properties, normalize=normalize, datadirs=sample_data)
    elif packetqueue:
        trace_generator = TraceGeneratorPacketQueue(link_properties, normalize=normalize)
    elif codel:
        trace_generator = TraceGeneratorCodel(link_properties, normalize=normalize, base_interval=10, codel_threshold=5)
    else:
        trace_generator = TraceGeneratorByteQueue(link_properties, normalize=normalize)

    if data_seed:
        np.random.seed(data_seed)
    trace_sample = trace_generator.generate_trace_sample(seq_len)
    input_v, output_v = trace_generator.feature_vector_from_sample(trace_sample)

    # load the model
    model = torch.jit.load(model_file)
    model.eval()

    # run the inputs through the model
    batch_size = 1
    hidden = torch.zeros(num_layers, batch_size, hidden_size)
    if use_lstm:
        hidden = (hidden, torch.zeros(num_layers, batch_size, hidden_size))
    dataX = torch.tensor(input_v, dtype=torch.float32).unsqueeze(dim=0)
    dataY = torch.tensor(output_v, dtype=torch.float32).unsqueeze(dim=0)
    trace_generator.print_means(dataX, dataY)
    print(f"dataX: {dataX.shape}\tdataY: {dataY.shape}")
    backlog_pred, dropped_pred, _ = model(dataX, hidden)
    print(f"backlog_pred: {backlog_pred.shape}\tdropped_pred: {dropped_pred.shape}")

    # reshape the outputs
    dropped_pred_binary = torch.argmax(dropped_pred, dim=2)
    dropped_pred_softbinary = torch.softmax(dropped_pred, dim=2)
    true_backlog = dataY[:, :, 0].squeeze().numpy()
    true_drops = dataY[:,:,1].squeeze().numpy()
    predicted_backlog = backlog_pred.squeeze().detach().numpy()
    predicted_drops = dropped_pred_binary.squeeze().detach().numpy()

    # make a plot
    pkt_arrival_times_v = trace_sample.pkt_arrival_times_v[0:seq_len]

    if normalize:
        true_backlog *= 1000
        predicted_backlog *= 1000

    # turn off interactive mode so plots don't display until we call plt.show()
    plt.ioff()

    plt.figure(figsize=(12, 6))
    print(f"pkt_arrival_times_v: {pkt_arrival_times_v.shape}\ttrue_backlog: {true_backlog.shape}")
    plt.plot(pkt_arrival_times_v, true_backlog, label="Generated Backlog", color='green', linewidth=2.5, zorder=1)
    plt.plot(pkt_arrival_times_v, predicted_backlog, label="Predicted Backlog", linestyle="dashed", color='red',
             linewidth=2.5, zorder=1)

    # Real dropped packet positions
    drop_indices_real = np.where(true_drops == 1)[0]
    plt.scatter(pkt_arrival_times_v[drop_indices_real], true_backlog[drop_indices_real], color='blue', marker='x',
                label="Real Dropped Packets", linewidth=2.5, zorder=2)

    # Predicted dropped packet positions
    # drop_indices_pred = np.argmax(predicted_drops, axis=-1)
    # drop_indices_pred = np.where(drop_indices_pred == 1)[0]
    drop_indices_pred = np.where(predicted_drops == 1)[0]
    print(f"pkt_arrival_times_v: {pkt_arrival_times_v.shape}\tpredicted_backlog: {predicted_backlog.shape}")
    plt.scatter(pkt_arrival_times_v[drop_indices_pred], predicted_backlog[drop_indices_pred], color='orange',
                marker='o',
                label="Predicted Dropped Packets", linewidth=2, zorder=2)

    plt.xlabel("Time [ms]", fontsize=18)
    plt.ylabel("Backlog [KByte]", fontsize=18)
    # plt.title("Generated vs Predicted Backlog and Dropped Packets")
    plt.legend()
    plt.grid()
    plt.draw()
    # tell matplotlib we are done with the figure
    #plt.close()

    # ============ Drop Position Match Accuracy ============ #
    # Get sets of real and predicted dropped positions
    # pred_dropped_status = np.argmax(predicted_drops, axis=-1)
    print_stats = True
    if print_stats:
        pred_dropped_status = predicted_drops.astype(int)
        real_dropped_status = true_drops.astype(int)
        pred_indices = set(np.where(pred_dropped_status == 1)[0])
        real_indices = set(np.where(real_dropped_status == 1)[0])
        matched = pred_indices & real_indices
        correct = len(matched)
        total = len(real_indices)
        accuracy = correct / total * 100 if total > 0 else 0.0

        pred_indices = [xx.item() for xx in pred_indices]
        real_indices = [xx.item() for xx in real_indices]
        matched = [xx.item() for xx in matched]
        print(f"Drop position match accuracy: {accuracy:.2f}% ({correct}/{total})")
        #print("Ground truth dropped positions:", sorted(real_indices))
        #print("Predicted dropped positions:", sorted(pred_indices))
        #rint("Correctly predicted positions:", sorted(matched))
        print(f"Number of drops:  real={len(real_indices)}  predicted={len(pred_indices)}")

        tensor_w1 = torch.from_numpy(real_dropped_status.astype(dtype=np.float64))
        tensor_w2 = torch.from_numpy(pred_dropped_status.astype(dtype=np.float64))
        print("\n\nWasserstein Results:")
        emp1_loss = torch.sum(stats_loss.symmetric_earthmover(tensor_w1, tensor_w2, p=1, normalize=True))
        emp2_loss = torch.sum(stats_loss.symmetric_earthmover(tensor_w1, tensor_w2, p=2, normalize=True))
        print(f"Symmetric earthmover loss p1: {emp1_loss}\tp2: {emp2_loss}")
        print("Wasserstein loss", stats_loss.torch_wasserstein_loss(tensor_w1, tensor_w2).data,
              stats_loss.torch_wasserstein_loss(tensor_w1, tensor_w2).requires_grad)
        print("Energy loss", stats_loss.torch_energy_loss(tensor_w1, tensor_w2).data,
              stats_loss.torch_wasserstein_loss(tensor_w1, tensor_w2).requires_grad)
        print("p == 1.5 CDF loss", stats_loss.torch_cdf_loss(tensor_w1, tensor_w2, p=1.5).data)
        print("Validate Checking Errors:", stats_loss.torch_validate_distibution(tensor_w1, tensor_w2))
    plt.show()



# ======================================
# ======================================
# ======================================

if __name__ == "__main__":
    main()
