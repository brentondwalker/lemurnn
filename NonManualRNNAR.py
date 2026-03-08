import torch
import torch.nn as nn
from LinkEmuModelAR import LinkEmuModelAR


class NonManualRNNAR(LinkEmuModelAR):
    def __init__(self, input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, loadpath=None,
                 nonlinearity='relu', dropout_rate=0.0, use_deltas=False):
        self.model_name = f"drop{nonlinearity}rnn_ar{"d" if use_deltas else ""}"
        self.nonlinearity = nonlinearity

        # Define the size of the autoregressive feedback (1 backlog + 2 dropped)
        self.ar_size = 4 if use_deltas else 3

        super(NonManualRNNAR, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                             learning_rate=learning_rate, dropout_rate=dropout_rate, loadpath=loadpath,
                                             use_deltas=use_deltas)

        # CRITICAL CHANGE: Increase the RNN input size to accept the external inputs + the previous outputs
        self.rnn = nn.RNN(input_size=self.input_size + self.ar_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, nonlinearity=self.nonlinearity, batch_first=True)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.fc = nn.Linear(in_features=self.hidden_size, out_features=3)

    def forward(self, x, hidden, ar_x=None, teacher_forcing_ratio=0.0):
        if self.use_deltas:
            return self.forward_deltas(x, hidden, ar_x=ar_x, teacher_forcing_ratio=teacher_forcing_ratio)
        return self.forward_std(x, hidden, ar_x=ar_x, teacher_forcing_ratio=teacher_forcing_ratio)

    def forward_deltas(self, x, hidden, ar_x=None, teacher_forcing_ratio=0.0):
        batch_size, seq_len, _ = x.size()
        device = x.device

        ###print(f"forward_deltas()  ar_x: {ar_x.shape}")
        backlog_outputs = []
        dropped_outputs = []

        # ar_x needs to be shape (batch, 1, 4) at every step
        curr_ar_input = torch.zeros(batch_size, 1, self.ar_size, device=device)

        # We need to track the absolute backlog manually during the slow path
        prev_cumulative = torch.zeros(batch_size, 1, 1, device=device)

        for t in range(seq_len):
            curr_x = x[:, t:t + 1, :]
            rnn_input = torch.cat([curr_x, curr_ar_input], dim=-1)
            ###print(f"curr_x: {curr_x.shape}\t curr_ar_input: {curr_ar_input.shape}\t rnn_input: {rnn_input.shape}")

            out, hidden = self.rnn(rnn_input, hidden)
            combined_out = self.fc(out)

            # The model predicts the DELTA, not the absolute value!
            pred_delta = combined_out[:, :, 0:1]
            dropped_t = combined_out[:, :, 1:3]

            # --- THE RESIDUAL CONNECTION ---
            if ar_x is not None and teacher_forcing_ratio == 1.0:
                # In 100% Teacher Forcing, we trust the ground-truth previous cumulative from ar_x
                # Assuming ar_x feature index 1 is the cumulative backlog
                curr_cumulative = ar_x[:, t:t + 1, 1:2] + pred_delta
            else:
                # In inference/slow path, we add the predicted delta to our running total
                curr_cumulative = prev_cumulative + pred_delta

            # Physics constraint: Backlog cannot be negative.
            curr_cumulative = torch.relu(curr_cumulative)

            backlog_outputs.append(curr_cumulative)
            dropped_outputs.append(dropped_t)

            # Decide the next Autoregressive Input
            use_teacher_forcing = (ar_x is not None) and (torch.rand(1).item() < teacher_forcing_ratio)

            if use_teacher_forcing:
                curr_ar_input = ar_x[:, t:t + 1, :]
                prev_cumulative = ar_x[:, t:t + 1, 1:2]  # Update our tracker for the next step
            else:
                # ---------------------------------------------------------
                # THE FIX: Always use Hard 1-Hot for Autoregressive Feedback
                # ---------------------------------------------------------
                # Whether training or evaluating, the model must see the exact
                # same binary state transitions it will see in production.
                dropped_idx = torch.argmax(dropped_t, dim=-1, keepdim=True)
                hard_drop_pred = torch.zeros_like(dropped_t).scatter_(-1, dropped_idx, 1.0)

                # We detach the entire AR input to prevent BPTT from looping back
                # through the inputs, which causes exploding gradients. The loss
                # function will still handle the gradients for the drop accuracy!
                curr_ar_input = torch.cat([pred_delta, curr_cumulative, hard_drop_pred], dim=-1).detach()
                prev_cumulative = curr_cumulative.detach()

        # Stack lists back into tensors
        backlog_out = torch.cat(backlog_outputs, dim=1)
        dropped_out = torch.cat(dropped_outputs, dim=1)

        return backlog_out, dropped_out, hidden


    def forward_std(self, x, hidden, ar_x=None, teacher_forcing_ratio=0.0):
        """
        x: (batch, seq_len, 4) - Standard external inputs
        ar_x: (batch, seq_len, 3) - Shifted ground truth outputs (for teacher forcing)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # ---------------------------------------------------------
        # FAST PATH: 100% Teacher Forcing (Parallelized)
        # ---------------------------------------------------------
        if ar_x is not None and teacher_forcing_ratio == 1.0:
            # Concatenate features along the last dimension: (batch, seq, 4) + (batch, seq, 3) -> (batch, seq, 7)
            rnn_input = torch.cat([x, ar_x], dim=-1)
            out, hidden = self.rnn(rnn_input, hidden)

            if self.dropout_rate > 0:
                out = self.dropout(out)

            combined_out = self.fc(out)
            return combined_out[:, :, 0:1], combined_out[:, :, 1:3], hidden

        # ---------------------------------------------------------
        # SLOW PATH: Autoregressive Loop / Scheduled Sampling
        # ---------------------------------------------------------
        backlog_outputs = []
        dropped_outputs = []

        # Initial autoregressive input (defaults to zeros for the very first time step)
        curr_ar_input = torch.zeros(batch_size, 1, self.ar_size, device=device)

        for t in range(seq_len):
            curr_x = x[:, t:t + 1, :]  # Slice out the current time step: (batch, 1, 4)
            rnn_input = torch.cat([curr_x, curr_ar_input], dim=-1)  # (batch, 1, 7)

            out, hidden = self.rnn(rnn_input, hidden)
            if self.dropout_rate > 0:
                out = self.dropout(out)

            combined_out = self.fc(out)

            backlog_t = combined_out[:, :, 0:1]  # (batch, 1, 1)
            dropped_t = combined_out[:, :, 1:3]  # (batch, 1, 2)

            backlog_outputs.append(backlog_t)
            dropped_outputs.append(dropped_t)

            # Decide the next Autoregressive Input
            use_teacher_forcing = (ar_x is not None) and (torch.rand(1).item() < teacher_forcing_ratio)

            if use_teacher_forcing:
                curr_ar_input = ar_x[:, t:t + 1, :]
            else:
                # ---------------------------------------------------------
                # NON-DELTAS AUTOREGRESSIVE FEEDBACK
                # ---------------------------------------------------------
                # 1. The backlog prediction is already the absolute value
                # (backlog_t = combined_out[:, :, 0:1] from earlier in the loop)

                # 2. Convert the raw drop logits into a hard 1-hot vector
                dropped_idx = torch.argmax(dropped_t, dim=-1, keepdim=True)
                hard_drop_pred = torch.zeros_like(dropped_t).scatter_(-1, dropped_idx, 1.0)

                # 3. Concatenate and detach to prevent exploding gradients
                # The gradient will NOT flow backward through this input,
                # but the loss function still gets the raw logits to train on!
                curr_ar_input = torch.cat([backlog_t, hard_drop_pred], dim=-1).detach()

        # Stack lists back into tensors
        backlog_out = torch.cat(backlog_outputs, dim=1)
        dropped_out = torch.cat(dropped_outputs, dim=1)

        return backlog_out, dropped_out, hidden

    def get_model_name(self):
        return super().get_model_name()

    def new_instance(self):
        return self.__class__(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              learning_rate=self.learning_rate, nonlinearity=self.nonlinearity,
                              dropout_rate=self.dropout_rate, use_deltas=self.use_deltas)

    def new_hidden_tensor(self, batch_size, device=None):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def get_extra_model_properties(self):
        return {
            'dropout_rate': self.dropout_rate,
            'nonlinearity': self.nonlinearity
        }

    def load_extra_model_properties(self, model_properties):
        self.dropout_rate = model_properties.get('dropout_rate', 0.0)
        self.nonlinearity = model_properties.get('nonlinearity', 'relu')
