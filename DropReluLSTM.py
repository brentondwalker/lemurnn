import torch
import torch.nn as nn
import torch.nn.functional as F

from LinkEmuModel import LinkEmuModel


class ReluLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        gates = self.W_x(x) + self.W_h(h_prev)
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        g = F.relu(g_gate) # ReLU instead of tanh

        c = f * c_prev + i * g
        h = o * F.relu(c)
        return h, c


class DropReluLSTM(LinkEmuModel):
    def __init__(self, input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, loadpath=None, dropout_rate=0.0):
        self.model_name = "droprelulstm"
        super(DropReluLSTM, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                           learning_rate=learning_rate, dropout_rate=dropout_rate, loadpath=loadpath)
        self.cell = ReluLSTMCell(input_size, hidden_size)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(hidden_size, 1 + 2)  # backlog + 2-class dropped

    def forward(self, x, hidden):
        batch_size, seq_length, _ = x.size()
        h, c = hidden

        outputs = []
        for t in range(seq_length):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # [batch, seq, hidden]
        combined_out = self.fc(outputs)

        backlog_out = combined_out[:, :, 0:1]  # [batch, seq, 1]
        dropped_out = combined_out[:, :, 1:3]  # [batch, seq, 2]
        return backlog_out, dropped_out, (h, c)


    def new_instance(self):
        """
        Override this because our constructor accepts more args than the base class.
        And these things (dropout and activation) have to be available in the constructor.
        There must be a more elegant way to handle this.
        :return:
        """
        return self.__class__(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              learning_rate=self.learning_rate,
                              dropout_rate=self.dropout_rate)

    def new_hidden_tensor(self, batch_size, device=None):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device), torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def get_extra_model_properties(self):
        extra_model_properties = {
            'dropout_rate': self.dropout_rate,
        }
        return extra_model_properties

    def load_extra_model_properties(self, model_properties):
        self.dropout_rate = model_properties.get('dropout_rate', 0.0)
        return

