import dataclasses
import json
import random
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_stats_loss as stats_loss
from LatencyPredictor import LatencyPredictor, TrainingRecord, GradientTracker
from LinkEmuModel import LinkEmuModel
from TraceGenerator import TraceGenerator


class LatencyPredictorTTS(LatencyPredictor):
    """
    A LatencyPredictor is given a packet arrival time and packet size,
    and predicts its latency and drop status.
    This one uses Temporal Target Smearing (TTS) to compute loss for packet drops,
    as opposed to earthmover distance that I was using earlier.
    """

    trainer_name = "trainer_tts"

    model_type = 'rnntts'

    def __init__(self, model: LinkEmuModel, trace_generator: TraceGenerator,
                 device=None, seed=None, loadpath=None, track_grad=False,
                 drop_masking=False, wandb_run=None, tb_chunk_size=None, tts_window=15):
        """
        Use Temporal Target Smearing as a metric to compare drop predictions.
        """
        self.earthmover_p = 1
        self.tts_window_size =  tts_window # The smearing window for TTS
        super().__init__(model, trace_generator, device=device, seed=seed, loadpath=loadpath, track_grad=track_grad,
                         drop_masking=drop_masking, wandb_run=wandb_run, tb_chunk_size=tb_chunk_size)

    def get_extra_model_properties(self):
        extra_model_properties = {
            'earthmover_p': self.earthmover_p,
            'tts_window_size': self.tts_window_size
        }
        return extra_model_properties

    def load_extra_model_properties(self, model_properties):
        self.earthmover_p = model_properties.get('earthmover_p', 1)
        self.tts_window_size = model_properties.get('tts_window_size', 5)
        return

    def get_smeared_drops(self, y_drops, window_size=5):
        """
        Applies Temporal Target Smearing (TTS) using a 1D Convolution.
        y_drops: Shape (batch, seq_len)
        """
        batch_size, seq_len = y_drops.size()
        y_drops_conv = y_drops.unsqueeze(1).float()

        # NEW: Sharper decay so neighbors are at most 0.5
        # Example for window_size=5: [0.0, 0.5, 1.0, 0.5, 0.0]
        center = window_size // 2
        kernel = torch.tensor([max(0.0, 1.0 - abs(i - center) * 0.5) for i in range(window_size)])
        kernel = kernel.view(1, 1, -1).to(y_drops.device)

        smeared_targets = F.conv1d(y_drops_conv, kernel, padding=center)
        smeared_targets = torch.clamp(smeared_targets.squeeze(1), max=1.0)

        return smeared_targets

    def train(self, n_epochs=1, loss_file=None, ads_loss_interval=0):
        self.set_training_directory(create=True)
        self.model.save_model_properties()
        previous_lr = self.model.optimizer.param_groups[0]['lr']
        training_log_filename = f"{self.training_directory}/training_log.dat"
        training_history_filename = f"{self.training_directory}/training_history.json"

        criterion_backlog = nn.L1Loss()
        criterion_dropped = nn.CrossEntropyLoss()
        testmodel = self.model.new_instance().to(self.device)

        scheduler = ReduceLROnPlateau(
            self.model.optimizer,
            mode='min',
            factor=0.5,
            patience=15,
            min_lr=1e-6
        )

        ads_loss = {0: 0, 1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
        grad_tracker_backlog = GradientTracker('backlog', self.training_directory, track_grad=self.track_grad)
        grad_tracker_dropped = GradientTracker('dropped', self.training_directory, track_grad=self.track_grad)
        grad_tracker_droprate = GradientTracker('droprate', self.training_directory, track_grad=self.track_grad)
        # Replaced emp tracker with tts tracker
        grad_tracker_tts = GradientTracker('tts', self.training_directory, track_grad=self.track_grad)
        ads_new_model = False

        output_scale = 1.0
        if self.trace_generator.normalize:
            output_scale = self.trace_generator.link_properties.max_pkt_size
        normalize_earthmover = True

        test_loss, t_backlog_loss, t_backlog_loss_n, t_dropped_loss = 0, 0, 0, 0
        t_dropped_em1_loss, t_dropped_em2_loss, t_dropped_em15_loss, t_dropped_emp_loss = 0, 0, 0, 0
        t_droprate_loss, t_tts_loss = 0.0, 0.0
        ads_str = "\t".join(["0.0000"] * 6)
        test_set_losses = {
            f"test-{tt}": {'total_loss': 0, 'backlog_loss': 0, 'dropped_loss': 0, 'em1_loss': 0, 'tts_loss': 0} for tt
            in self.trace_generator.test_traffic_types}

        for epoch_i in range(n_epochs):
            self.epoch += 1
            grad_tracker_backlog.clear()
            grad_tracker_dropped.clear()
            grad_tracker_droprate.clear()
            grad_tracker_tts.clear()
            new_best_model = False
            self.model.train()
            train_loss, train_backlog_loss, train_dropped_loss, train_droprate_loss, train_tts_loss = 0, 0, 0, 0, 0
            num_train_steps = 0
            batch_size = 0

            train_loaders = list(self.trace_generator.get_loader_iterator('train'))
            active_iterators = [iter(loader) for loader in train_loaders]

            while active_iterators:
                idx = random.randrange(len(active_iterators))

                try:
                    X_batch, y_batch = next(active_iterators[idx])
                    batch_size, seq_length, _ = X_batch.size()
                    num_train_steps += 1

                    hidden = self.model.new_hidden_tensor(batch_size, self.device)
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    chunk_size = self.tb_chunk_size if self.tb_chunk_size is not None else seq_length
                    for i in range(0, seq_length, chunk_size):
                        X_chunk = X_batch[:, i:i + chunk_size, :]
                        y_chunk = y_batch[:, i:i + chunk_size, :]

                        if isinstance(hidden, tuple):
                            hidden = tuple(h.detach() for h in hidden)
                        else:
                            hidden = hidden.detach()

                        backlog_pred, dropped_pred, hidden = self.model(X_chunk, hidden)
                        backlog_target = y_chunk[:, :, 0].unsqueeze(-1)
                        dropped_target = y_chunk[:, :, 1].long()

                        dropped_pred_binary = torch.softmax(dropped_pred, dim=2)[:, :, 1]

                        if self.drop_masking:
                            backlog_target = backlog_target * (1 - dropped_target.unsqueeze(dim=-1))
                            backlog_pred = backlog_pred * (1 - dropped_target.unsqueeze(dim=-1))

                        backlog_loss = criterion_backlog(backlog_pred, backlog_target)
                        dropped_loss = criterion_dropped(dropped_pred.view(-1, 2), dropped_target.view(-1))

                        droprate_loss = torch.mean(
                            torch.abs(torch.sum(y_chunk[:, :, 1], dim=1) - torch.sum(dropped_pred_binary, dim=1)))

                        # --- TEMPORAL TARGET SMEARING (TTS) ---
                        dropped_target_smeared = self.get_smeared_drops(y_chunk[:, :, 1].float(),
                                                                        window_size=self.tts_window_size)
                        tts_loss = torch.mean(torch.abs(dropped_target_smeared - dropped_pred_binary))

                        #loss = backlog_loss + dropped_loss + tts_loss
                        loss = backlog_loss + droprate_loss + tts_loss

                        train_loss += loss.item()
                        train_backlog_loss += backlog_loss.item()
                        train_dropped_loss += dropped_loss.item()
                        train_droprate_loss += droprate_loss.item()
                        train_tts_loss += tts_loss.item()

                        self.model.optimizer.zero_grad()
                        grad_tracker_backlog.add(backlog_loss, self.model)
                        grad_tracker_dropped.add(dropped_loss, self.model)
                        grad_tracker_droprate.add(droprate_loss, self.model)
                        grad_tracker_tts.add(tts_loss, self.model)

                        self.model.optimizer.zero_grad()
                        loss.backward()
                        self.model.optimizer.step()

                except StopIteration:
                    active_iterators.pop(idx)

            train_loss /= num_train_steps
            train_backlog_loss /= num_train_steps
            train_dropped_loss /= num_train_steps
            train_droprate_loss /= num_train_steps
            train_tts_loss /= num_train_steps

            train_loss_details = {'backlog_loss': train_backlog_loss,
                                  'dropped_loss': train_dropped_loss,
                                  'droprate_loss': train_droprate_loss,
                                  'train_tts_loss': train_tts_loss}

            grad_tracker_backlog.write(self.epoch, num_samples=num_train_steps)
            grad_tracker_dropped.write(self.epoch, num_samples=num_train_steps)
            grad_tracker_droprate.write(self.epoch, num_samples=num_train_steps)
            grad_tracker_tts.write(self.epoch, num_samples=num_train_steps)

            # Validation step
            self.model.eval()
            val_loss, v_backlog_loss, v_dropped_loss, v_droprate_loss, v_tts_loss = 0, 0, 0, 0, 0
            with torch.no_grad():
                loader = self.trace_generator.get_loader('val')
                for X_val, y_val in loader:
                    batch_size_val, seq_length, _ = X_val.size()
                    hidden = self.model.new_hidden_tensor(batch_size_val, self.device)

                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    backlog_target_val = y_val[:, :, 0].unsqueeze(-1)
                    dropped_target_val = y_val[:, :, 1].long()

                    backlog_pred_val, dropped_pred_val, _ = self.model(X_val, hidden)
                    dropped_pred_val_binary = torch.argmax(dropped_pred_val, dim=2)
                    dropped_pred_val_prob = torch.softmax(dropped_pred_val, dim=2)[:, :, 1]

                    if self.drop_masking:
                        backlog_target_val = backlog_target_val * (1 - dropped_target_val.unsqueeze(dim=-1))
                        backlog_pred_val = backlog_pred_val * (1 - dropped_target_val.unsqueeze(dim=-1))

                    val_backlog_loss = criterion_backlog(backlog_pred_val * output_scale,
                                                         backlog_target_val * output_scale)
                    val_dropped_loss = criterion_dropped(dropped_pred_val.view(-1, 2), dropped_target_val.view(-1))

                    val_droprate_loss = torch.mean(
                        torch.abs(torch.sum(y_val[:, :, 1], dim=1) - torch.sum(dropped_pred_val_binary, dim=1)))

                    # Compute TTS for Validation instead of EMD
                    dropped_target_val_smeared = self.get_smeared_drops(y_val[:, :, 1].float(),
                                                                        window_size=self.tts_window_size)
                    val_tts_loss_step = torch.mean(torch.abs(dropped_target_val_smeared - dropped_pred_val_prob))

                    #val_loss += (val_backlog_loss + val_tts_loss_step).item()
                    val_loss += (val_backlog_loss + val_droprate_loss + val_tts_loss_step).item()
                    v_backlog_loss += val_backlog_loss.item()
                    v_dropped_loss += val_dropped_loss.item()
                    v_droprate_loss += val_droprate_loss.item()
                    v_tts_loss += val_tts_loss_step.item()

            num_val_steps = len(loader)
            val_loss /= num_val_steps
            v_backlog_loss /= num_val_steps
            v_dropped_loss /= num_val_steps
            v_droprate_loss /= num_val_steps
            v_tts_loss /= num_val_steps

            if val_loss < self.best_loss:
                print("*!*!*!* Found a new best model: ", self.epoch, val_loss)
                self.best_loss = val_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.best_model_epoch = self.epoch
                self.model.save_model_state(self.epoch, wandb_run=self.wandb_run)
                new_best_model = True
                ads_new_model = True

            val_loss_details = {'backlog_loss': v_backlog_loss,
                                'dropped_loss': v_dropped_loss,
                                'droprate_loss': v_droprate_loss,
                                'tts_loss': v_tts_loss}

            # evaluate against test set ONLY if we found a new best model!
            if new_best_model:
                test_loss, t_backlog_loss, t_backlog_loss_n, t_dropped_loss = 0, 0, 0, 0
                t_dropped_em1_loss, t_dropped_em2_loss, t_dropped_em15_loss, t_dropped_emp_loss = 0, 0, 0, 0
                t_droprate_loss, t_tts_loss = 0.0, 0.0
                if ads_loss_interval > 0 and ads_new_model and (self.epoch % ads_loss_interval) == 0:
                    ads_loss = {0: 0, 1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
                test_set_losses = {}
                testmodel.load_state_dict(self.best_model)
                testmodel.eval()

                with torch.no_grad():
                    total_num_test_steps = 0
                    for traffic_type in self.trace_generator.test_traffic_types:
                        testp_loss, tp_backlog_loss, tp_backlog_loss_n, tp_dropped_loss = 0, 0, 0, 0
                        tp_dropped_em1_loss, tp_dropped_em2_loss, tp_dropped_em15_loss, tp_dropped_emp_loss = 0, 0, 0, 0
                        tp_droprate_loss, tp_tts_loss = 0.0, 0.0
                        test_dataset_name = f"test-{traffic_type}"
                        loader = self.trace_generator.get_loader(test_dataset_name)
                        for X_test, y_test in loader:
                            batch_size_test, seq_length, _ = X_test.size()
                            hidden = self.model.new_hidden_tensor(batch_size_test, self.device)

                            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                            backlog_target_test = y_test[:, :, 0].unsqueeze(-1)
                            dropped_target_test = y_test[:, :, 1].long()

                            backlog_pred_test, dropped_pred_test, _ = testmodel(X_test, hidden)
                            if self.drop_masking:
                                backlog_target_test = backlog_target_test * (1 - dropped_target_test.unsqueeze(dim=-1))
                                backlog_pred_test = backlog_pred_test * (1 - dropped_target_test.unsqueeze(dim=-1))

                            backlog_loss_test = criterion_backlog(backlog_pred_test * output_scale,
                                                                  backlog_target_test * output_scale)
                            backlog_loss_test_n = criterion_backlog(
                                backlog_pred_test / X_test[:, :, 2].unsqueeze(dim=-1),
                                backlog_target_test / X_test[:, :, 2].unsqueeze(dim=-1))
                            dropped_loss_test = criterion_dropped(dropped_pred_test.view(-1, 2),
                                                                  dropped_target_test.view(-1))

                            dropped_pred_test_binary = torch.argmax(dropped_pred_test, dim=2)
                            dropped_pred_test_prob = torch.softmax(dropped_pred_test, dim=2)[:, :, 1]

                            # Track all the original EMD metrics for backwards compatibility
                            tp_backlog_loss += backlog_loss_test.item()
                            tp_backlog_loss_n += backlog_loss_test_n.item()
                            tp_dropped_loss += dropped_loss_test.item()
                            tp_dropped_em1_loss += torch.mean(
                                stats_loss.symmetric_earthmover(y_test[:, :, 1], dropped_pred_test_binary, p=1,
                                                                normalize=False)).item()
                            tp_dropped_em2_loss += torch.mean(
                                stats_loss.symmetric_earthmover(y_test[:, :, 1], dropped_pred_test_binary, p=2,
                                                                normalize=False)).item()
                            tp_dropped_em15_loss += torch.mean(
                                stats_loss.symmetric_earthmover(y_test[:, :, 1], dropped_pred_test_binary, p=1.5,
                                                                normalize=False)).item()
                            tp_dropped_emp_loss += torch.mean(
                                stats_loss.symmetric_earthmover(y_test[:, :, 1], dropped_pred_test_binary,
                                                                p=self.earthmover_p, normalize=False)).item()

                            tp_droprate_loss += torch.mean(torch.abs(
                                torch.sum(y_test[:, :, 1], dim=1) - torch.sum(dropped_pred_test_binary, dim=1))).item()

                            # Track the new TTS metric alongside them
                            dropped_target_test_smeared = self.get_smeared_drops(y_test[:, :, 1].float(),
                                                                                 window_size=self.tts_window_size)
                            tp_tts_loss += torch.mean(
                                torch.abs(dropped_target_test_smeared - dropped_pred_test_prob)).item()

                            if ads_loss_interval > 0 and ads_new_model and (self.epoch % ads_loss_interval) == 0:
                                for i in range(batch_size_test):
                                    ads_loss[0] += self.adropsim(dropped_pred_test_binary[i, :], y_test[i, :, 1], 0)
                                    radius = 1
                                    for p in range(5):
                                        ads_loss[radius] += self.adropsim(dropped_pred_test_binary[i, :],
                                                                          y_test[i, :, 1], radius)
                                        radius *= 2

                        t_backlog_loss += tp_backlog_loss
                        t_backlog_loss_n += tp_backlog_loss_n
                        t_dropped_loss += tp_dropped_loss
                        t_dropped_em1_loss += tp_dropped_em1_loss
                        t_dropped_em2_loss += tp_dropped_em2_loss
                        t_dropped_em15_loss += tp_dropped_em15_loss
                        t_dropped_emp_loss += tp_dropped_emp_loss
                        t_droprate_loss += tp_droprate_loss
                        t_tts_loss += tp_tts_loss

                        num_test_steps = len(loader)
                        total_num_test_steps += num_test_steps
                        tp_backlog_loss /= (num_test_steps * 64)
                        tp_backlog_loss_n /= (num_test_steps * 64)
                        tp_dropped_loss /= (num_test_steps * 64)
                        tp_dropped_em1_loss /= num_test_steps
                        tp_dropped_em2_loss /= num_test_steps
                        tp_dropped_em15_loss /= num_test_steps
                        tp_dropped_emp_loss /= num_test_steps
                        tp_droprate_loss /= num_test_steps
                        tp_tts_loss /= num_test_steps

                        # Use TTS for the final reported test_loss
                        test_p_loss = tp_backlog_loss + tp_tts_loss
                        test_set_losses[test_dataset_name] = {
                            'total_loss': test_p_loss, 'backlog_loss': tp_backlog_loss,
                            'dropped_loss': tp_dropped_loss, 'em1_loss': tp_dropped_em1_loss,
                            'tts_loss': tp_tts_loss
                        }

                    t_backlog_loss /= (total_num_test_steps * 64)
                    t_backlog_loss_n /= (total_num_test_steps * 64)
                    t_dropped_loss /= (total_num_test_steps * 64)
                    t_dropped_em1_loss /= total_num_test_steps
                    t_dropped_em2_loss /= total_num_test_steps
                    t_dropped_em15_loss /= total_num_test_steps
                    t_dropped_emp_loss /= total_num_test_steps
                    t_droprate_loss /= total_num_test_steps
                    t_tts_loss /= total_num_test_steps

                    test_loss = t_backlog_loss + t_tts_loss

                self.prediction_plot(test_index=0, data_set_name='test', display_plot=False, save_png=True,
                                     print_stats=False, file_suffix=f"_epoch{self.epoch}")

            if ads_loss_interval > 0 and ads_new_model and (self.epoch % ads_loss_interval) == 0:
                ads_loss[0] /= num_test_steps
                radius = 1
                for p in range(5):
                    ads_loss[radius] /= num_test_steps
                    radius *= 2
            ads_str = "\t".join(
                f"{x:.4f}" for x in (ads_loss[0], ads_loss[1], ads_loss[2], ads_loss[4], ads_loss[8], ads_loss[16]))

            test_loss_details = {'backlog_loss': t_backlog_loss,
                                 'backlog_loss_n': t_backlog_loss_n,
                                 'dropped_loss': t_dropped_loss,
                                 'droprate_loss': t_droprate_loss,
                                 'earthmover_loss': t_dropped_emp_loss,
                                 'tts_loss': t_tts_loss,
                                 'ads_loss': ads_loss}

            print(f"Epoch {self.epoch + 1}: Train: {train_loss:.4f} , Val: {val_loss:.4f} , Test: {test_loss:.4f}")
            print(
                f"\tTBLoss: {t_backlog_loss:.4f}, TBLossN: {t_backlog_loss_n:.4f}, TDLoss: {t_dropped_loss:.4f} , TDRLoss: {t_droprate_loss:.4f} , TTTS: {t_tts_loss:.4f} , TDEM1: {t_dropped_em1_loss:.4f} , TDEM2: {t_dropped_em2_loss:.4f} , TDEM15: {t_dropped_em15_loss:.4f}")
            print(f"\tTADSLoss: {ads_str}")
            print(f"\tTDroprateLoss: {t_droprate_loss}")
            for traffic_type in self.trace_generator.test_traffic_types:
                test_dataset_name = f"test-{traffic_type}"
                loss_rec = test_set_losses[test_dataset_name]
                print(
                    f"\t{test_dataset_name}\tLoss: {loss_rec['total_loss']:.4f}\tBLoss: {loss_rec['backlog_loss']:.4f}\tDLoss: {loss_rec['dropped_loss']:.4f}\tTTSLoss: {loss_rec['tts_loss']:.4f}\tEM1Loss: {loss_rec['em1_loss']:.4f}")
            if self.track_grad:
                print("\n".join([xx.get_str(num_samples=num_train_steps) for xx in
                                 [grad_tracker_backlog, grad_tracker_dropped, grad_tracker_droprate,
                                  grad_tracker_tts]]))

            # Appended t_tts_loss right before the ads_str to preserve parsing structure
            with open(training_log_filename, "a", buffering=1) as loss_file:
                loss_file.write(
                    f"{self.epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{test_loss:.4f}\t{self.best_loss:.4f}\t{t_backlog_loss:.4f}\t{t_backlog_loss_n:.4f}\t{t_dropped_loss:.4f}\t{t_dropped_em1_loss:.4f}\t{t_dropped_em2_loss:.4f}\t{t_dropped_em15_loss:.4f}\t{t_droprate_loss:.4f}\t{t_tts_loss:.4f}\t{ads_str}\t{self.best_model_epoch}\n")

            scheduler.step(val_loss)

            current_lr = self.model.optimizer.param_groups[0]['lr']
            if current_lr < previous_lr:
                print("\n******************************************************")
                print(f"\n📉 Validation loss plateaued! Dropping learning rate to {current_lr:.6e}\n")
                previous_lr = current_lr

            self.training_history.append(TrainingRecord(self.epoch, current_lr, self.best_model_file,
                                                        train_loss, val_loss, test_loss,
                                                        train_loss_details, val_loss_details, test_loss_details))

            with open(training_history_filename, "a", buffering=1) as history_file:
                history_file.write(json.dumps(dataclasses.asdict(self.training_history[-1])))
                history_file.write("\n")

            if self.wandb_run:
                wandb_log_data = {"epoch": self.epoch,
                                  "train_loss": train_loss,
                                  "val_loss": val_loss,
                                  "test_loss": test_loss,
                                  "best_loss": self.best_loss,
                                  "t_backlog_loss": t_backlog_loss,
                                  "t_dropped_loss": t_dropped_loss,
                                  "t_tts_loss": t_tts_loss,
                                  "t_dropped_em1_loss": t_dropped_em1_loss,
                                  "t_dropped_em2_loss": t_dropped_em2_loss,
                                  "t_dropped_em15_loss": t_dropped_em15_loss,
                                  "t_droprate_loss": t_droprate_loss,
                                  "ads_str": ads_str,
                                  "best_model_epoch": self.best_model_epoch}
                for traffic_type in self.trace_generator.test_traffic_types:
                    test_dataset_name = f"test-{traffic_type}"
                    loss_rec = test_set_losses[test_dataset_name]
                    wandb_log_data[f"{test_dataset_name}_loss"] = loss_rec['total_loss']
                    wandb_log_data[f"{test_dataset_name}_backlog_loss"] = loss_rec['backlog_loss']
                    wandb_log_data[f"{test_dataset_name}_dropped_loss"] = loss_rec['dropped_loss']
                    wandb_log_data[f"{test_dataset_name}_tts_loss"] = loss_rec['tts_loss']
                    wandb_log_data[f"{test_dataset_name}_em1_loss"] = loss_rec['em1_loss']
                self.wandb_run.log(wandb_log_data)