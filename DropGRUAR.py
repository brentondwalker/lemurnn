import torch
import torch.nn as nn
from LinkEmuModel import LinkEmuModel
from LinkEmuModelAR import LinkEmuModelAR


class DropGRUAR(LinkEmuModelAR):
    def __init__(self, input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, loadpath=None, dropout_rate=0.0):
        self.model_name = f"dropgru_ar"
        self.nonlinearity = "tanh"
        # Define the size of the autoregressive feedback (1 backlog + 2 dropped)
        self.ar_size = 3
        super(DropGRUAR, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                           learning_rate=learning_rate, dropout_rate=dropout_rate, loadpath=loadpath)
        # after calling super(), the internal data fields will be populated.
        # either from the arguments passed in, or from loading a saved state
        # CRITICAL CHANGE: Increase the GRU input size to accept the external inputs + the previous outputs
        self.gru = nn.GRU(input_size=self.input_size + self.ar_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        # 1 for backlog + 2 for dropped classification
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=3)

    def forward(self, x, hidden, ar_x=None, teacher_forcing_ratio=0.0):
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
            gru_input = torch.cat([x, ar_x], dim=-1)
            out, hidden = self.gru(gru_input, hidden)

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
            gru_input = torch.cat([curr_x, curr_ar_input], dim=-1)  # (batch, 1, 7)

            out, hidden = self.gru(gru_input, hidden)
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
                # Format model's own prediction to match the (batch, 1, 3) shape
                if self.training:
                    # Differentiable approximation for training
                    dropped_pred = torch.softmax(dropped_t, dim=-1)
                else:
                    # Hard one-hot for pure inference
                    dropped_idx = torch.argmax(dropped_t, dim=-1, keepdim=True)
                    dropped_pred = torch.zeros_like(dropped_t).scatter_(-1, dropped_idx, 1.0)

                # Concatenate the real backlog value and the classification prediction, then detach
                curr_ar_input = torch.cat([backlog_t, dropped_pred], dim=-1).detach()

        # Stack lists back into tensors
        backlog_out = torch.cat(backlog_outputs, dim=1)
        dropped_out = torch.cat(dropped_outputs, dim=1)

        return backlog_out, dropped_out, hidden

    def new_instance(self):
        return self.__class__(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              learning_rate=self.learning_rate, dropout_rate=self.dropout_rate)

    def new_hidden_tensor(self, batch_size, device=None):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def get_extra_model_properties(self):
        extra_model_properties = {
            'dropout_rate': self.dropout_rate,
            'nonlinearity': self.nonlinearity
        }
        return extra_model_properties

    def load_extra_model_properties(self, model_properties):
        self.dropout_rate = model_properties.get('dropout_rate', 0.0)
        self.nonlinearity = model_properties.get('nonlinearity', 'tanh')
        return

