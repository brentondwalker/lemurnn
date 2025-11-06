import dataclasses
import json
import time
from copy import deepcopy
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_stats_loss as stats_loss

from LinkEmuModel import LinkEmuModel
from TraceGenerator import TraceGenerator



class GradientTracker:

    def __init__(self, name="unnamed", training_directory=None, track_grad=True):
        self.track_grad = track_grad   # can disable the tracker
        self.name = name
        self.sum = 0.0
        self.grads = None
        self.epoch = 0
        self.training_directory = None
        self.filename = None
        self.epoch = 0

        if training_directory:
            if os.path.isdir(training_directory):
                self.training_directory = training_directory
                self.filename = f"{self.training_directory}/grad-tracker-{self.name}.dat"
            else:
                print(f"ERROR: GradientTracker: directory does not exist: {training_directory}")

    def clear(self):
        self.grads = None
        self.sum = 0.0

    def add(self, loss, model:LinkEmuModel):
        if self.track_grad:
            gtree = torch.autograd.grad(outputs=loss, inputs=model.parameters(), retain_graph=True)
            if gtree:
                gl = [torch.linalg.norm(xx) for xx in gtree]
                if not self.grads:
                    self.grads = [g.item() for g in gl]
                else:
                    self.grads = [g1 + g2.item() for g1, g2 in zip(self.grads, gl)]
                self.sum = sum(self.grads)

    def write(self, epoch=None, num_samples=1):
        if self.track_grad:
            if epoch:
                self.epoch = epoch
            if self.filename:
                with open(self.filename, "a") as loss_file:
                    # no nested f-strings in python older than 3.12
                    grad_string = "\t".join([f"{(x/num_samples):.4f}" for x in self.grads])
                    loss_file.write(f"{self.epoch}\t{(self.sum/num_samples):.4f}\t{grad_string}\n")
            self.epoch += 1

    def get_str(self, num_samples=1):
        grad_string = ""
        if self.grads:
            grad_string = "\t".join([f"{(x/num_samples):.4f}" for x in self.grads])
        return f"{self.name}\t{(self.sum/num_samples):.4f}\t{grad_string}"


@dataclass
class TrainingRecord:
    epoch:int
    learning_rate:float
    best_model:str
    train_loss:float
    val_loss:float
    test_loss:float
    train_loss_details:dict
    val_loss_details:dict
    test_loss_details:dict


class LatencyPredictor:
    """
    A LatencyPredictor is given a packet arrival time and packet size,
    and predicts its latency and drop status.
    """
    trainer_name = "trainer_bx"

    def __init__(self, model:LinkEmuModel, trace_generator:TraceGenerator,
                 device=None, seed=None, loadpath=None, track_grad=False):
        """
        XXX in the case of loadpath, we should load all the model info from the
        :param trace_generator:
        :param device:
        :param seed:
        :param loadpath:
        """
        if device:
            self.device = device
        else:
            self.device = self.get_device()
        self.seed = seed
        if self.seed:
            torch.manual_seed(self.seed)
        self.epoch = -1
        self.track_grad = track_grad
        self.trace_generator = trace_generator
        self.input_size = trace_generator.input_size()
        self.output_size = trace_generator.output_size()
        self.model:LinkEmuModel = model.to(self.device)
        if loadpath:
            self.model.load_model_state(loadpath, self.device)
        self.best_model = deepcopy(self.model.state_dict())
        self.best_model_file: str = None
        self.best_model_epoch = -1
        self.best_loss = np.inf
        self.set_data_directory(os.getcwd())
        self.training_directory = None
        self.set_training_directory(create=True)
        self.save_link_properties()
        self.trace_generator.save_dataset_properties(f"{self.training_directory}/dataset-properties.json")

        self.training_history: List[TrainingRecord] = []
        self.data_directory = None


    def save_link_properties(self):
        link_properties_filename = f"{self.training_directory}/link-properties.json"
        with open(link_properties_filename, "w") as link_properties_file:
                link_properties_file.write(json.dumps(dataclasses.asdict(self.trace_generator.link_properties)))
                link_properties_file.write("\n")

    def set_data_directory(self, path):
        if os.path.isdir(path):
            self.data_directory = path
        else:
            print(f"ERROR: directory does not exist: {path}")

    def set_training_directory(self, path=None, create=False):
        if path:
            if os.path.isdir(path) or create:
                self.training_directory = path
            else:
                print(f"ERROR: training directory does not exist: {path}")
        elif not self.training_directory:
                self.training_directory = f"{self.data_directory}/model-training/model-{self.trainer_name}-{self.model.get_model_name()}-{self.trace_generator.get_dataset_string()}-l{self.model.num_layers}_h{self.model.hidden_size}-{int(time.time())}"
        if create:
            if os.path.isdir(self.training_directory):
                print(f"WARNING: training dir already exists: {self.training_directory}")
            else:
                os.makedirs(self.training_directory, exist_ok=True)
        if self.model:
            self.model.set_training_directory(self.training_directory)

    def get_device(self):
        # Check if GPU is available
        if torch.cuda.is_available():
            device = torch.device("cuda")  # Use the first GPU
            print("GPU is available.")
        else:
            device = torch.device("cpu")  # Use CPU
            print("Using CPU.")
        return device

    def weight_string(self, weight_dict):
        weights = [entry for arr in weight_dict.items() for entry in arr[1].cpu().ravel().numpy().tolist()]
        return "\t".join([str(ww) for ww in weights])


    def train(self, learning_rate=0.001, n_epochs=1, loss_file=None, ads_loss_interval=0):
        self.set_training_directory(create=True)
        self.learning_rate = learning_rate
        self.model.save_model_properties()
        training_log_filename = f"{self.training_directory}/training_log.dat"
        training_history_filename = f"{self.training_directory}/training_history.json"

        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion_backlog = nn.L1Loss()
        criterion_dropped = nn.CrossEntropyLoss()
        #testmodel = NonManualRNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)
        testmodel = self.model.new_instance().to(self.device)
        ads_loss = {0: 0, 1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
        ads_new_model = False

        # if we normalized inputs and outpus for training, we still want to
        # compute the val and test loss on their absolute values
        output_scale = 1.0
        if self.trace_generator.normalize:
            output_scale = self.trace_generator.link_properties.max_pkt_size

        for epoch_i in range(n_epochs):
            self.epoch += 1
            new_best_model = False
            self.model.train()  # Set to training mode
            train_loss, train_backlog_loss, train_dropped_loss, train_droprate_loss, train_wasserstein_loss = 0, 0, 0, 0, 0
            #loader = self.trace_generator.get_loader('train')
            num_train_samples = 0
            for loader in self.trace_generator.get_loader_iterator('train'):
                for X_batch, y_batch in loader:
                    #print(X_batch.shape, y_batch.shape)
                    batch_size, seq_length, _ = X_batch.size()
                    #hidden = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size).to(self.device)  # Move hidden to same device
                    hidden = self.model.new_hidden_tensor(batch_size, self.device)

                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    backlog_pred, dropped_pred, hidden = self.model(X_batch, hidden)  # Forward pass
                    backlog_target = y_batch[:, :, 0].unsqueeze(-1)  # Shape: [batch_size, seq_length, 1]
                    dropped_target = y_batch[:, :, 1].long()  # Shape: [batch_size, seq_length] (for CrossEntropyLoss)
                    dropped_pred_binary = torch.softmax(dropped_pred, dim=2)[:, :, 1]
                    backlog_loss = criterion_backlog(backlog_pred, backlog_target)
                    dropped_loss = criterion_dropped(dropped_pred.view(-1, 2), dropped_target.view(-1))
                    droprate_loss = torch.sum(
                        torch.abs(torch.sum(y_batch[:, :, 1], dim=1) - torch.sum(dropped_pred_binary, dim=1)))
                    wasserstein_loss = stats_loss.torch_wasserstein_loss(y_batch[:, :, 1], dropped_pred_binary)  # .data

                    #loss = backlog_loss + dropped_loss + droprate_loss + wasserstein_loss
                    loss = backlog_loss + dropped_loss
                    train_loss += loss.item()
                    train_backlog_loss += backlog_loss.item()
                    train_dropped_loss += dropped_loss.item()
                    train_droprate_loss += droprate_loss.item()
                    train_wasserstein_loss += wasserstein_loss.item()

                    self.model.optimizer.zero_grad()  # Zero gradients
                    loss.backward()  # Backpropagation
                    self.model.optimizer.step()  # Update parameters

                num_train_samples += len(loader) * batch_size
            train_loss /= num_train_samples
            train_backlog_loss /= num_train_samples
            train_dropped_loss /= num_train_samples
            train_droprate_loss /= num_train_samples
            train_wasserstein_loss /= num_train_samples

            train_loss_details = {'backlog_loss': train_backlog_loss,
                                  'dropped_loss': train_dropped_loss,
                                  'droprate_loss': train_droprate_loss,
                                  'wasserstein_loss': train_wasserstein_loss}

            # Validation step
            self.model.eval()
            val_loss, v_backlog_loss, v_dropped_loss, v_droprate_loss, v_wasserstein_loss = 0, 0, 0, 0, 0
            with torch.no_grad():
                loader = self.trace_generator.get_loader('val')
                for X_val, y_val in loader:
                    batch_size_val, _, _ = X_val.size()
                    #hidden = torch.zeros(self.model.num_layers, batch_size_val, self.model.hidden_size).to(self.device)
                    hidden = self.model.new_hidden_tensor(batch_size_val, self.device)

                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    backlog_target_val = y_val[:, :, 0].unsqueeze(-1)
                    dropped_target_val = y_val[:, :, 1].long()
                    backlog_pred_val, dropped_pred_val, _ = self.model(X_val, hidden)
                    dropped_pred_val_binary = torch.argmax(dropped_pred_val, dim=2)

                    val_backlog_loss = criterion_backlog(backlog_pred_val * output_scale, backlog_target_val * output_scale)
                    val_dropped_loss = criterion_dropped(dropped_pred_val.view(-1, 2), dropped_target_val.view(-1))
                    val_droprate_loss = torch.sum(
                        torch.abs(torch.sum(y_val[:, :, 1], dim=1) - torch.sum(dropped_pred_val_binary, dim=1)))
                    val_wasserstein_loss = stats_loss.torch_wasserstein_loss(y_val[:, :, 1],
                                                                             dropped_pred_val_binary)  #.data

                    val_loss += (val_backlog_loss + val_dropped_loss + val_droprate_loss + val_wasserstein_loss).item()
                    v_backlog_loss += val_backlog_loss.item()
                    v_dropped_loss += val_dropped_loss.item()
                    v_droprate_loss += val_droprate_loss.item()
                    v_wasserstein_loss += val_wasserstein_loss.item()

            num_val_samples = len(loader) * batch_size_val
            val_loss /= num_val_samples
            v_backlog_loss /= num_val_samples
            v_dropped_loss /= num_val_samples
            v_droprate_loss /= num_val_samples
            v_wasserstein_loss /= num_val_samples # XXX not done in notebook

            # Check if the current model is the best
            if val_loss < self.best_loss:
                print("*!*!*!* Found a new best model: ", self.epoch, val_loss)
                self.best_loss = val_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.best_model_epoch = self.epoch
                self.model.save_model_state(self.epoch)
                new_best_model = True
                ads_new_model = True

            val_loss_details = {'backlog_loss': v_backlog_loss,
                                  'dropped_loss': v_dropped_loss,
                                  'droprate_loss': v_droprate_loss,
                                  'wasserstein_loss': v_wasserstein_loss}

            # evaluate against test set using the current best model!!
            test_loss, t_backlog_loss, t_backlog_loss_n, t_dropped_loss = 0, 0, 0, 0
            t_dropped_wa_loss, t_dropped_en_loss, t_dropped_p15_loss = 0, 0, 0
            t_droprate_loss = 0.0
            if ads_loss_interval > 0 and ads_new_model and (self.epoch % ads_loss_interval) == 0:
                ads_loss = {0: 0, 1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
            testmodel.load_state_dict(self.best_model)  # Load the current best model
            testmodel.eval()  # Ensure evaluation mode

            with torch.no_grad():
                loader = self.trace_generator.get_loader('test')
                for X_test, y_test in loader:
                    batch_size_test, _, _ = X_test.size()
                    #hidden = torch.zeros(self.model.num_layers, batch_size_test, self.model.hidden_size).to(self.device)
                    hidden = self.model.new_hidden_tensor(batch_size_test, self.device)

                    X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                    backlog_target_test = y_test[:, :, 0].unsqueeze(-1)
                    dropped_target_test = y_test[:, :, 1].long()

                    backlog_pred_test, dropped_pred_test, _ = testmodel(X_test, hidden)

                    backlog_loss_test = criterion_backlog(backlog_pred_test * output_scale, backlog_target_test * output_scale)
                    # index of capacity input is currently 2
                    backlog_loss_test_n = criterion_backlog(backlog_pred_test/X_test[:,:,2].unsqueeze(dim=-1), backlog_target_test/X_test[:,:,2].unsqueeze(dim=-1))
                    dropped_loss_test = criterion_dropped(dropped_pred_test.view(-1, 2), dropped_target_test.view(-1))
                    dropped_pred_test_binary = torch.argmax(dropped_pred_test, dim=2)

                    t_backlog_loss += backlog_loss_test.item()
                    t_backlog_loss_n += backlog_loss_test_n.item()
                    t_dropped_loss += dropped_loss_test.item()
                    t_dropped_wa_loss += stats_loss.torch_wasserstein_loss(y_test[:, :, 1],
                                                                           dropped_pred_test_binary).item()  #.data
                    t_dropped_en_loss += stats_loss.torch_energy_loss(y_test[:, :, 1], dropped_pred_test_binary).item()  #.data
                    t_dropped_p15_loss += stats_loss.torch_cdf_loss(y_test[:, :, 1], dropped_pred_test_binary,
                                                                    p=1.5).item()  #.data
                    t_droprate_loss += torch.sum(torch.abs(
                        torch.sum(y_test[:, :, 1], dim=1) - torch.sum(dropped_pred_test_binary, dim=1))).item()
                    #print(dropped_pred_test_binary.shape, y_test[0, :, 1].shape)
                    if ads_loss_interval > 0 and ads_new_model and (self.epoch % ads_loss_interval) == 0:
                        # Be frugal with this, because I have not parallelized it.
                        # It is super slow.
                        for i in range(batch_size_test):
                            ads_loss[0] += self.adropsim(dropped_pred_test_binary[i,:], y_test[i, :, 1], 0)
                            radius = 1
                            for p in range(5):
                                ads_loss[radius] += self.adropsim(dropped_pred_test_binary[i,:], y_test[i, :, 1], radius)
                                radius *= 2
                            #print(f"{self.epoch}.{i}\t{ads_loss}")

            num_test_samples = len(loader) * batch_size_test
            t_backlog_loss /= num_test_samples
            t_backlog_loss_n /= num_test_samples
            t_dropped_loss /= num_test_samples
            t_dropped_wa_loss /= num_test_samples
            t_dropped_en_loss /= num_test_samples
            t_dropped_p15_loss /= num_test_samples
            t_droprate_loss /= num_test_samples
            test_loss = t_backlog_loss + t_dropped_loss + t_droprate_loss + t_dropped_wa_loss
            if new_best_model:
                self.prediction_plot(test_index=0, data_set_name='test', display_plot=False, save_png=True, print_stats=False, file_suffix=f"_epoch{self.epoch}")
            if ads_loss_interval > 0 and ads_new_model and (self.epoch % ads_loss_interval) == 0:
                ads_loss[0] /= num_test_samples
                radius = 1
                for p in range(5):
                    ads_loss[radius] /= num_test_samples
                    radius *= 2
            ads_str = "\t".join(f"{x:.4f}" for x in (ads_loss[0], ads_loss[1], ads_loss[2], ads_loss[4], ads_loss[8], ads_loss[16]))

            test_loss_details = {'backlog_loss': t_backlog_loss,
                                 'backlog_loss_n': t_backlog_loss_n,
                                 'dropped_loss': t_dropped_loss,
                                 'droprate_loss': t_droprate_loss,
                                 'wasserstein_loss': t_dropped_wa_loss,
                                 'ads_loss': ads_loss}

            print(f"Epoch {self.epoch + 1}: Train: {train_loss:.4f} , Val: {val_loss:.4f} , Test: {test_loss:.4f}")
            print(f"\tTBLoss: {t_backlog_loss:.4f}, TBLossN: {t_backlog_loss_n:.4f}, TDLoss: {t_dropped_loss:.4f} , TDWA {t_dropped_wa_loss:.4f} , TDEN: {t_dropped_en_loss:.4f} , TDP15: {t_dropped_p15_loss:.4f}")
            print(f"\tTADSLoss: {ads_str}")
            print(f"\tTDroprateLoss: {t_droprate_loss}")

            # get the current model parameters
            with open(training_log_filename, "a", buffering=1) as loss_file:
                loss_file.write(
                    f"{self.epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{test_loss:.4f}\t{self.best_loss:.4f}\t{t_backlog_loss:.4f}\t{t_backlog_loss_n:.4f}\t{t_dropped_loss:.4f}\t{t_dropped_wa_loss:.4f}\t{t_dropped_en_loss:.4f}\t{t_dropped_p15_loss:.4f}\t{t_droprate_loss:.4f}\t{ads_str}\t{self.best_model_epoch}\n")

            self.training_history.append(TrainingRecord(self.epoch, self.learning_rate, self.best_model_file,
                train_loss, val_loss, test_loss,
                train_loss_details, val_loss_details, test_loss_details))
            with open(training_history_filename, "a", buffering=1) as history_file:
                history_file.write(json.dumps(dataclasses.asdict(self.training_history[-1])))
                history_file.write("\n")


    def adropsim(self, s1, s2, drop_radius=0):
        """
        Additive drop sequence similarity.
        This has the linear property: D(s1+s2, s1) = D(s2, 0)
        """
        seq_len = len(s1)
        w1 = np.zeros(seq_len + 2 * drop_radius)
        w2 = np.zeros(seq_len + 2 * drop_radius)
        for i in range(seq_len):
            wi = i + drop_radius
            if s1[i]:
                for j in range(-drop_radius, drop_radius + 1):
                    w1[wi + j] += 1.0 / (2 * drop_radius + 1)
            if s2[i]:
                for j in range(-drop_radius, drop_radius + 1):
                    w2[wi + j] += 1.0 / (2 * drop_radius + 1)
        ww = w1 - w2
        result = np.sum(np.abs(ww))
        return result


    def predict_sample(self, model_dict=None, test_index=0, data_set_name='test', print_stats=True):
        """
        Use the current best model, or whatever is passed in, to generate predictions
        from a single sample (of the test set).

        :param model_dict:
        :param test_index:
        :return:
        """

        # first retrieve the data for this test_index
        input_features, output_features = self.trace_generator.get_feature_vector(test_index=test_index, data_set_name=data_set_name)
        dataX  = torch.tensor(input_features, dtype=torch.float32).unsqueeze(dim=0)
        dataY = torch.tensor(output_features, dtype=torch.float32).unsqueeze(dim=0)

        # allocate a model to use for eval
        #eval_model = NonManualRNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)  #.to(self.device)
        eval_model = self.model.new_instance()  #.to(self.device)
        if model_dict:
            eval_model.load_state_dict(model_dict)
        else:
            eval_model.load_state_dict(self.best_model)
        eval_model.eval()

        wa_dist, wasoft_dist, en_dist, ensoft_dist, p15_dist, p15soft_dist = 0,0,0,0,0,0

        with torch.no_grad():
            #hidden = torch.zeros(self.model.num_layers, dataX.size(0), self.model.hidden_size)  #.to(self.device)
            hidden = self.model.new_hidden_tensor(dataX.size(0))
            backlog_pred, dropped_pred, _ = eval_model(dataX, hidden)
            dropped_pred_binary = torch.argmax(dropped_pred, dim=2)
            dropped_pred_softbinary = torch.softmax(dropped_pred, dim=2)

            if print_stats:
                wa_dist += stats_loss.torch_wasserstein_loss(dataY[:, :, 1], dropped_pred_binary).data
                wasoft_dist += stats_loss.torch_wasserstein_loss(dataY[:, :, 1], dropped_pred_softbinary[:,:,1]).data
                en_dist += stats_loss.torch_energy_loss(dataY[:, :, 1], dropped_pred_binary).data
                ensoft_dist += stats_loss.torch_energy_loss(dataY[:, :, 1], dropped_pred_softbinary[:,:,1]).data
                p15_dist += stats_loss.torch_cdf_loss(dataY[:, :, 1], dropped_pred_binary, p=1.5).data
                p15soft_dist += stats_loss.torch_cdf_loss(dataY[:, :, 1], dropped_pred_softbinary[:,:,1], p=1.5).data

        if print_stats:
            print("Wasserstein Loss Results: \n",
            "Wasserstein distance",wa_dist,"\n",
            "Wasserstein softmax distance",wasoft_dist,"\n",
            "Energy distance",en_dist,"\n",
            "Energy softmax distance",ensoft_dist,"\n",
            "p == 1.5 CDF loss",p15_dist,"\n",
            "p == 1.5 CDF softmax loss",p15soft_dist,"\n")

        return dataY[:, :, 0].squeeze().numpy(), dataY[:,:,1].squeeze().numpy(), backlog_pred.squeeze().numpy(), dropped_pred_binary.squeeze().numpy()


    def prediction_plot(self, test_index=0, data_set_name='test', display_plot=True, save_png=True, save_pdf=False, print_stats=True, file_suffix=""):
        """
        Given a bunch of predictions, visualize them.

        :param test_index:
        :param data_set_name:
        :return:
        """
        # ============ Visualization ============
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
            'legend.fontsize': 14,
            'legend.frameon': True,
            'lines.linewidth': 2,
            'pdf.fonttype': 42  # embed TrueType fonts for LaTeX compatibility
        })

        true_backlog, true_drops, predicted_backlog, predicted_drops = self.predict_sample(test_index=test_index, data_set_name=data_set_name, print_stats=print_stats)
        pkt_arrival_times_v = self.trace_generator.get_sample(test_index, data_set_name).pkt_arrival_times_v

        if self.trace_generator.normalize:
            true_backlog *= self.trace_generator.link_properties.max_pkt_size
            predicted_backlog *= self.trace_generator.link_properties.max_pkt_size

        # turn off interactive mode so plots don't display until we call plt.show()
        plt.ioff()

        plt.figure(figsize=(12, 6))
        plt.plot(pkt_arrival_times_v, true_backlog, label="Generated Backlog", color='green', linewidth=2.5, zorder=1)
        plt.plot(pkt_arrival_times_v, predicted_backlog, label="Predicted Backlog", linestyle="dashed", color='red', linewidth=2.5, zorder=1)

        # Real dropped packet positions
        drop_indices_real = np.where(true_drops == 1)[0]
        plt.scatter(pkt_arrival_times_v[drop_indices_real], true_backlog[drop_indices_real], color='blue', marker='x',
                    label="Real Dropped Packets", linewidth=2.5, zorder=2)

        # Predicted dropped packet positions
        #drop_indices_pred = np.argmax(predicted_drops, axis=-1)
        #drop_indices_pred = np.where(drop_indices_pred == 1)[0]
        drop_indices_pred = np.where(predicted_drops == 1)[0]
        plt.scatter(pkt_arrival_times_v[drop_indices_pred], predicted_backlog[drop_indices_pred], color='orange', marker='o',
                    label="Predicted Dropped Packets", linewidth=2, zorder=2)

        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Backlog", fontsize=18)
        # plt.title("Generated vs Predicted Backlog and Dropped Packets")
        plt.legend()
        plt.grid()
        if save_pdf:
            plt.savefig(f"{self.training_directory}/BD_plot_{data_set_name}_sample{test_index}{file_suffix}.pdf", format='pdf')
        if save_png:
            plt.savefig(f"{self.training_directory}/BD_plot_{data_set_name}_sample{test_index}{file_suffix}.png", format='png')
        if display_plot:
            plt.show()
        # tell matplotlib we are done with the figure
        plt.close()

        # ============ Drop Position Match Accuracy ============ #
        # Get sets of real and predicted dropped positions
        #pred_dropped_status = np.argmax(predicted_drops, axis=-1)
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
            print(f"Drop position match accuracy (example {test_index}): {accuracy:.2f}% ({correct}/{total})")
            print("Ground truth dropped positions:", sorted(real_indices))
            print("Predicted dropped positions:", sorted(pred_indices))
            print("Correctly predicted positions:", sorted(matched))
            print(f"Number of drops:  real={len(real_indices)}  predicted={len(pred_indices)}")
            print(f"adropsim(0) = {self.adropsim(pred_dropped_status, real_dropped_status, 0)}")
            print(f"adropsim(1) = {self.adropsim(pred_dropped_status, real_dropped_status, 1)}")
            print(f"adropsim(2) = {self.adropsim(pred_dropped_status, real_dropped_status, 2)}")
            print(f"adropsim(4) = {self.adropsim(pred_dropped_status, real_dropped_status, 4)}")
            print(f"adropsim(8) = {self.adropsim(pred_dropped_status, real_dropped_status, 8)}")
            print(f"adropsim(16) = {self.adropsim(pred_dropped_status, real_dropped_status, 16)}")

            tensor_w1 = torch.from_numpy(real_dropped_status.astype(dtype=np.float64))
            tensor_w2 = torch.from_numpy(pred_dropped_status.astype(dtype=np.float64))
            print("\n\nWasserstein Results:")
            print("Wasserstein loss", stats_loss.torch_wasserstein_loss(tensor_w1, tensor_w2).data,
                  stats_loss.torch_wasserstein_loss(tensor_w1, tensor_w2).requires_grad)
            print("Energy loss", stats_loss.torch_energy_loss(tensor_w1, tensor_w2).data,
                  stats_loss.torch_wasserstein_loss(tensor_w1, tensor_w2).requires_grad)
            print("p == 1.5 CDF loss", stats_loss.torch_cdf_loss(tensor_w1, tensor_w2, p=1.5).data)
            print("Validate Checking Errors:", stats_loss.torch_validate_distibution(tensor_w1, tensor_w2))



    def predict_dataset(self, loader, model_dict=None, print_stats=True):
        """
        Use the current best model, or whatever is passewd in to generate predictions
        from the input sequences in the loader.

        :param loader:
        :param model_dict:
        :return:
        """
        #eval_model = NonManualRNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)
        eval_model = self.model.new_instance().to(self.device)

        if model_dict:
            eval_model.load_state_dict(model_dict)
        else:
            eval_model.load_state_dict(self.best_model)

        eval_model.eval()

        predicted_backlogs = []
        real_backlogs = []
        predicted_drops = []
        real_drops = []
        predicted_drops_bin = []
        wa_dist, wasoft_dist, en_dist, ensoft_dist, p15_dist, p15soft_dist = 0,0,0,0,0,0

        with torch.no_grad():
            for X_test, y_test in loader:
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                #hidden = torch.zeros(self.model.num_layers, X_test.size(0), self.model.hidden_size).to(self.device)
                hidden = self.model.new_hidden_tensor(X_test.size(0))
                backlog_pred_test, dropped_pred_test, _ = eval_model(X_test, hidden)
                dropped_pred_test_binary = torch.argmax(dropped_pred_test, dim=2)
                dropped_pred_test_softbinary = torch.softmax(dropped_pred_test, dim=2)
                BATCH, _ = dropped_pred_test_binary.shape
                wa_dist += stats_loss.torch_wasserstein_loss(y_test[:, :, 1], dropped_pred_test_binary).data / BATCH
                wasoft_dist += stats_loss.torch_wasserstein_loss(y_test[:, :, 1], dropped_pred_test_softbinary[:,:,1]).data / BATCH
                en_dist += stats_loss.torch_energy_loss(y_test[:, :, 1], dropped_pred_test_binary).data / BATCH
                ensoft_dist += stats_loss.torch_energy_loss(y_test[:, :, 1], dropped_pred_test_softbinary[:,:,1]).data / BATCH
                p15_dist += stats_loss.torch_cdf_loss(y_test[:, :, 1], dropped_pred_test_binary, p=1.5).data / BATCH
                p15soft_dist += stats_loss.torch_cdf_loss(y_test[:, :, 1], dropped_pred_test_softbinary[:,:,1], p=1.5).data / BATCH

                predicted_backlogs.append(backlog_pred_test.cpu().numpy())
                predicted_drops.append(dropped_pred_test.cpu().numpy())

                real_backlogs.append(y_test[:, :, 0].cpu().numpy())
                real_drops.append(y_test[:, :, 1].cpu().numpy())

        num_batches = len(loader)
        if print_stats:
            print("Wasserstein Loss Results: \n",
            "Wasserstein distance",wa_dist/num_batches,"\n",
            "Wasserstein softmax distance",wasoft_dist/num_batches,"\n",
            "Energy distance",en_dist/num_batches,"\n",
            "Energy softmax distance",ensoft_dist/num_batches,"\n",
            "p == 1.5 CDF loss",p15_dist/num_batches,"\n",
            "p == 1.5 CDF softmax loss",p15soft_dist/num_batches,"\n")

        # Convert predictions to numpy arrays
        predicted_backlogs = np.concatenate(predicted_backlogs, axis=0)
        real_backlogs = np.concatenate(real_backlogs, axis=0)
        predicted_drops = np.concatenate(predicted_drops, axis=0)
        real_drops = np.concatenate(real_drops, axis=0)
        #print(predicted_drops.shape, real_drops.shape)
        #print(predicted_drops)
        return real_backlogs, real_drops, predicted_backlogs, predicted_drops

