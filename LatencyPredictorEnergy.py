import dataclasses
import json
from copy import deepcopy
import torch
import torch.nn as nn
import pytorch_stats_loss as stats_loss

from LatencyPredictor import LatencyPredictor, TrainingRecord
from LinkEmuModel import LinkEmuModel
from TraceGenerator import TraceGenerator


class LatencyPredictorEnergy(LatencyPredictor):
    """
    A LatencyPredictor is given a packet arrival time and packet size,
    and predicts its latency and drop status.
    """

    model_type = 'rnnenergy'

    def __init__(self, model:LinkEmuModel, trace_generator: TraceGenerator, device=None, seed=None, loadpath=None):
        """
        Because the superclass init() already creates the training directory and saves the model properties,
        we would have to set any variables we want before calling the super().init().
        Which seems like bad practice.
        So model_type and energy_distance_scale are hard coded in the class, which may be even worse.
        """
        self.energy_distance_scale = 10
        super().__init__(model, trace_generator, device=device, seed=seed, loadpath=loadpath)


    def get_extra_model_properties(self):
        extra_model_properties = {
            'energy_distance_scale': self.energy_distance_scale
        }
        return extra_model_properties

    def load_extra_model_properties(self, model_properties):
        self.energy_distance_scale = model_properties['energy_distance_scale']
        return


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
        testmodel = self.model.new_instance()
        ads_loss = {0: 0, 1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
        ads_new_model = False

        for epoch_i in range(n_epochs):
            self.epoch += 1
            new_best_model = False
            self.model.train()  # Set to training mode
            train_loss, train_backlog_loss, train_dropped_loss, train_droprate_loss, train_energy_loss = 0, 0, 0, 0, 0
            for X_batch, y_batch in self.trace_generator.train_loader:
                #print(X_batch.shape, y_batch.shape)
                batch_size, seq_length, _ = X_batch.size()
                hidden = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size).to(self.device)  # Move hidden to same device

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                backlog_pred, dropped_pred, hidden = self.model(X_batch, hidden.to(self.device))  # Forward pass
                backlog_target = y_batch[:, :, 0].unsqueeze(-1)  # Shape: [batch_size, seq_length, 1]
                dropped_target = y_batch[:, :, 1].long()  # Shape: [batch_size, seq_length] (for CrossEntropyLoss)
                dropped_pred_binary = torch.softmax(dropped_pred, dim=2)[:, :, 1]
                backlog_loss = criterion_backlog(backlog_pred, backlog_target)
                dropped_loss = criterion_dropped(dropped_pred.view(-1, 2), dropped_target.view(-1))
                droprate_loss = torch.sum(
                    torch.abs(torch.sum(y_batch[:, :, 1], dim=1) - torch.sum(dropped_pred_binary, dim=1)))
                wasserstein_loss = stats_loss.torch_wasserstein_loss(y_batch[:, :, 1], dropped_pred_binary)  # .data
                #energy_loss = self.energy_distance_scale * stats_loss.torch_energy_loss(y_batch[:, :, 1], dropped_pred_binary)
                #energy_loss = torch.tensor(10.0)
                #if backlog_loss < 1000:
                energy_loss = stats_loss.torch_cdf_loss_protected(y_batch[:, :, 1], dropped_pred_binary, p=2, normalize=False)  #/10.0 #, scaler=2.0)
                #print("-------------")
                #print(f"backlog_loss={backlog_loss:.4f}\tdroprate_loss={droprate_loss:.4f}\tenergy_loss={energy_loss:.4f}\twasserstein_loss={wasserstein_loss:.4f}")
                #print(type(backlog_loss), type(droprate_loss), type(energy_loss))
                #print(f"energy_loss: {energy_loss}")
                #print(f"2.0*energy_loss: {2.0*energy_loss}")

                loss = backlog_loss + droprate_loss + energy_loss
                #loss = backlog_loss + droprate_loss + dropped_loss + wasserstein_loss
                train_loss += loss.item()
                train_backlog_loss += backlog_loss.item()
                train_dropped_loss += dropped_loss.item()
                train_droprate_loss += droprate_loss.item()
                train_energy_loss += energy_loss.item()


                self.model.optimizer.zero_grad()  # Zero gradients
                loss.backward()  # Backpropagation
                self.model.optimizer.step()  # Update parameters

            num_train_samples = len(self.trace_generator.train_loader) * batch_size
            train_loss /= num_train_samples
            train_backlog_loss /= num_train_samples
            train_dropped_loss /= num_train_samples
            train_droprate_loss /= num_train_samples
            train_energy_loss /= num_train_samples

            train_loss_details = {'backlog_loss': train_backlog_loss,
                                  'dropped_loss': train_dropped_loss,
                                  'droprate_loss': train_droprate_loss,
                                  'train_energy_loss': train_energy_loss}

            # Validation step
            self.model.eval()
            val_loss, v_backlog_loss, v_dropped_loss, v_droprate_loss, v_energy_loss = 0, 0, 0, 0, 0
            with torch.no_grad():
                for X_val, y_val in self.trace_generator.val_loader:
                    batch_size_val, _, _ = X_val.size()
                    hidden = torch.zeros(self.model.num_layers, batch_size_val, self.model.hidden_size).to(self.device)

                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    backlog_target_val = y_val[:, :, 0].unsqueeze(-1)
                    dropped_target_val = y_val[:, :, 1].long()
                    backlog_pred_val, dropped_pred_val, _ = self.model(X_val, hidden)
                    dropped_pred_val_binary = torch.argmax(dropped_pred_val, dim=2)

                    val_backlog_loss = criterion_backlog(backlog_pred_val, backlog_target_val)
                    val_dropped_loss = criterion_dropped(dropped_pred_val.view(-1, 2), dropped_target_val.view(-1))
                    val_droprate_loss = torch.sum(
                        torch.abs(torch.sum(y_val[:, :, 1], dim=1) - torch.sum(dropped_pred_val_binary, dim=1)))
                    val_energy_loss = self.energy_distance_scale * stats_loss.torch_energy_loss(y_val[:, :, 1],
                                                                             dropped_pred_val_binary)  #.data

                    #val_loss += (val_backlog_loss + val_dropped_loss + val_droprate_loss + val_energy_loss).item()
                    val_loss += (val_backlog_loss + val_droprate_loss + val_energy_loss).item()
                    v_backlog_loss += val_backlog_loss.item()
                    v_dropped_loss += val_dropped_loss.item()
                    v_droprate_loss += val_droprate_loss.item()
                    v_energy_loss += val_energy_loss.item()

            num_val_samples = len(self.trace_generator.val_loader) * batch_size_val
            val_loss /= num_val_samples
            v_backlog_loss /= num_val_samples
            v_dropped_loss /= num_val_samples
            v_droprate_loss /= num_val_samples
            v_energy_loss /= num_val_samples # XXX not done in notebook

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
                                  'energy_loss': v_energy_loss}

            # evaluate against test set using the current best model!!
            test_loss, t_backlog_loss, t_backlog_loss_n, t_dropped_loss = 0, 0, 0, 0
            t_dropped_wa_loss, t_dropped_en_loss, t_dropped_p15_loss = 0, 0, 0
            t_droprate_loss = 0.0
            if ads_loss_interval > 0 and ads_new_model and (self.epoch % ads_loss_interval) == 0:
                ads_loss = {0: 0, 1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
            testmodel.load_state_dict(self.best_model)  # Load the current best model
            testmodel.eval()  # Ensure evaluation mode

            with torch.no_grad():
                for X_test, y_test in self.trace_generator.test_loader:
                    batch_size_test, _, _ = X_test.size()
                    hidden = torch.zeros(self.model.num_layers, batch_size_test, self.model.hidden_size).to(self.device)

                    X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                    backlog_target_test = y_test[:, :, 0].unsqueeze(-1)
                    dropped_target_test = y_test[:, :, 1].long()

                    backlog_pred_test, dropped_pred_test, _ = testmodel(X_test, hidden)

                    backlog_loss_test = criterion_backlog(backlog_pred_test, backlog_target_test)
                    # index of capacity input is currently 2
                    backlog_loss_test_n = criterion_backlog(backlog_pred_test/X_test[:,:,2].unsqueeze(dim=-1), backlog_target_test/X_test[:,:,2].unsqueeze(dim=-1))
                    dropped_loss_test = criterion_dropped(dropped_pred_test.view(-1, 2), dropped_target_test.view(-1))
                    dropped_pred_test_binary = torch.argmax(dropped_pred_test, dim=2)

                    t_backlog_loss += backlog_loss_test.item()
                    t_backlog_loss_n += backlog_loss_test_n.item()
                    t_dropped_loss += dropped_loss_test.item()
                    t_dropped_wa_loss += stats_loss.torch_wasserstein_loss(y_test[:, :, 1],
                                                                           dropped_pred_test_binary).item()  #.data
                    t_dropped_en_loss += self.energy_distance_scale * stats_loss.torch_energy_loss(y_test[:, :, 1], dropped_pred_test_binary).item()  #.data
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

            num_test_samples = len(self.trace_generator.test_loader) * batch_size_test
            t_backlog_loss /= num_test_samples
            t_backlog_loss_n /= num_test_samples
            t_dropped_loss /= num_test_samples
            t_dropped_wa_loss /= num_test_samples
            t_dropped_en_loss /= num_test_samples
            t_dropped_p15_loss /= num_test_samples
            t_droprate_loss /= num_test_samples
            test_loss = t_backlog_loss + t_droprate_loss + t_dropped_en_loss
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

