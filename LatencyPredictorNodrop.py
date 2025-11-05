import dataclasses
import json
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import pytorch_stats_loss as stats_loss
from LatencyPredictor import LatencyPredictor, TrainingRecord, GradientTracker
from LinkEmuModel import LinkEmuModel
from TraceGenerator import TraceGenerator


class LatencyPredictorNodrop(LatencyPredictor):
    """
    A LatencyPredictor is given a packet arrival time and packet size,
    and predicts its latency and drop status.
    """

    model_type = 'nodrop'

    def __init__(self, model:LinkEmuModel, trace_generator: TraceGenerator, device=None, seed=None, loadpath=None, track_grad=False):
        """
        Use earthmover distance as a metric to compare drop predictions.
        """
        super().__init__(model, trace_generator, device=device, seed=seed, loadpath=loadpath, track_grad=track_grad)


    def get_extra_model_properties(self):
        extra_model_properties = {}
        return extra_model_properties

    def load_extra_model_properties(self, model_properties):
        return

    def train(self, learning_rate=0.001, n_epochs=1, loss_file=None, ads_loss_interval=0):
        self.set_training_directory(create=True)
        self.learning_rate = learning_rate
        self.model.save_model_properties()
        training_log_filename = f"{self.training_directory}/training_log.dat"
        training_history_filename = f"{self.training_directory}/training_history.json"

        criterion_backlog = nn.L1Loss()
        testmodel = self.model.new_instance().to(self.device)
        ads_loss = {0: 0, 1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
        grad_tracker_backlog = GradientTracker('backlog', self.training_directory, track_grad=self.track_grad)
        for epoch_i in range(n_epochs):
            self.epoch += 1
            grad_tracker_backlog.clear()
            new_best_model = False
            self.model.train()  # Set to training mode
            train_loss, train_backlog_loss = 0, 0
            num_train_samples = 0
            for loader in self.trace_generator.get_loader_iterator('train'):
                for X_batch, y_batch in loader:
                    batch_size, seq_length, _ = X_batch.size()
                    hidden = self.model.new_hidden_tensor(batch_size, self.device)

                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    backlog_pred, hidden = self.model(X_batch, hidden)  # Forward pass
                    backlog_target = y_batch[:, :, 0].unsqueeze(-1)  # Shape: [batch_size, seq_length, 1]
                    backlog_loss = criterion_backlog(backlog_pred, backlog_target)

                    loss = backlog_loss
                    train_loss += loss.item()
                    train_backlog_loss += backlog_loss.item()

                    self.model.optimizer.zero_grad()  # Zero gradients
                    grad_tracker_backlog.add(backlog_loss, self.model)

                    self.model.optimizer.zero_grad()  # Zero gradients
                    loss.backward()  # Backpropagation
                    self.model.optimizer.step()  # Update parameters

                num_train_samples += len(loader) * batch_size
            train_loss /= num_train_samples
            train_backlog_loss /= num_train_samples

            train_loss_details = {'backlog_loss': train_backlog_loss}

            grad_tracker_backlog.write(self.epoch, num_samples=num_train_samples)

            # Validation step
            self.model.eval()
            val_loss, v_backlog_loss = 0, 0
            with torch.no_grad():
                loader = self.trace_generator.get_loader('val')
                for X_val, y_val in loader:
                    batch_size_val, _, _ = X_val.size()
                    hidden = self.model.new_hidden_tensor(batch_size_val, self.device)

                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    backlog_target_val = y_val[:, :, 0].unsqueeze(-1)
                    backlog_pred_val, _ = self.model(X_val, hidden)
                    val_backlog_loss = criterion_backlog(backlog_pred_val, backlog_target_val)
                    val_loss += val_backlog_loss.item()
                    v_backlog_loss += val_backlog_loss.item()

            num_val_samples = len(loader) * batch_size_val
            val_loss /= num_val_samples
            v_backlog_loss /= num_val_samples

            # Check if the current model is the best
            if val_loss < self.best_loss:
                print("*!*!*!* Found a new best model: ", self.epoch, val_loss)
                self.best_loss = val_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.best_model_epoch = self.epoch
                self.model.save_model_state(self.epoch)
                new_best_model = True
                ads_new_model = True

            val_loss_details = {'backlog_loss': v_backlog_loss}

            # evaluate against test set using the current best model!!
            test_loss, t_backlog_loss, t_backlog_loss_n = 0, 0, 0
            testmodel.load_state_dict(self.best_model)  # Load the current best model
            testmodel.eval()  # Ensure evaluation mode

            with torch.no_grad():
                loader = self.trace_generator.get_loader('test')
                for X_test, y_test in loader:
                    batch_size_test, _, _ = X_test.size()
                    hidden = self.model.new_hidden_tensor(batch_size_test, self.device)

                    X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                    backlog_target_test = y_test[:, :, 0].unsqueeze(-1)

                    backlog_pred_test, _ = testmodel(X_test, hidden)

                    backlog_loss_test = criterion_backlog(backlog_pred_test, backlog_target_test)
                    # index of capacity input is currently 2
                    backlog_loss_test_n = criterion_backlog(backlog_pred_test/X_test[:,:,0].unsqueeze(dim=-1), backlog_target_test/X_test[:,:,0].unsqueeze(dim=-1))

                    t_backlog_loss += backlog_loss_test.item()
                    t_backlog_loss_n += backlog_loss_test_n.item()

            num_test_samples = len(loader) * batch_size_test
            t_backlog_loss /= num_test_samples
            t_backlog_loss_n /= num_test_samples
            test_loss = t_backlog_loss
            if new_best_model:
                self.prediction_plot(test_index=0, data_set_name='test', display_plot=False, save_png=True, print_stats=False, file_suffix=f"_epoch{self.epoch}")

            test_loss_details = {'backlog_loss': t_backlog_loss,
                                 'backlog_loss_n': t_backlog_loss_n}

            print(f"Epoch {self.epoch + 1}: Train: {train_loss:.4f} , Val: {val_loss:.4f} , Test: {test_loss:.4f}")
            print(f"\tTBLoss: {t_backlog_loss:.4f}, TBLossN: {t_backlog_loss_n:.4f}")
            if self.track_grad:
                print("\n".join([xx.get_str(num_samples=num_train_samples) for xx in [grad_tracker_backlog]]))

            # get the current model parameters
            with open(training_log_filename, "a", buffering=1) as loss_file:
                loss_file.write(
                    f"{self.epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{test_loss:.4f}\t{self.best_loss:.4f}\t{t_backlog_loss:.4f}\t{t_backlog_loss_n:.4f}\t{self.best_model_epoch}\n")

            self.training_history.append(TrainingRecord(self.epoch, self.learning_rate, self.best_model_file,
                train_loss, val_loss, test_loss,
                train_loss_details, val_loss_details, test_loss_details))
            with open(training_history_filename, "a", buffering=1) as history_file:
                history_file.write(json.dumps(dataclasses.asdict(self.training_history[-1])))
                history_file.write("\n")


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
            backlog_pred, _ = eval_model(dataX, hidden)

        return dataY[:, :, 0].squeeze().numpy(), backlog_pred.squeeze().numpy()


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

        true_backlog, predicted_backlog = self.predict_sample(test_index=test_index, data_set_name=data_set_name, print_stats=print_stats)
        pkt_arrival_times_v = self.trace_generator.get_sample(test_index, data_set_name).pkt_arrival_times_v

        # turn off interactive mode so plots don't display until we call plt.show()
        plt.ioff()

        plt.figure(figsize=(12, 6))
        plt.plot(pkt_arrival_times_v, true_backlog, label="Generated Backlog", color='green', linewidth=2.5, zorder=1)
        plt.plot(pkt_arrival_times_v, predicted_backlog, label="Predicted Backlog", linestyle="dashed", color='red', linewidth=2.5, zorder=1)

        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Backlog", fontsize=18)
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
