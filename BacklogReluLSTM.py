import torch
import torch.nn as nn
import torch.nn.functional as F
from LinkEmuModel import LinkEmuModel

class Backlog_ReLU_LSTMCell(nn.Module):
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
        g = F.relu(g_gate)  # ReLU instead of tanh

        c = f * c_prev + i * g
        h = o * F.relu(c)
        return h, c

class BacklogReluLSTM(LinkEmuModel):
    #input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, loadpath=None
    def __init__(self, input_size=2, hidden_size=20, num_layers=1, learning_rate=0.001, loadpath=None):
        super(BacklogReluLSTM, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                           learning_rate=learning_rate, loadpath=loadpath)
        self.hidden_size = hidden_size
        self.cell = Backlog_ReLU_LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        batch_size, seq_length, _ = x.size()
        h, c = hidden
        outputs = []
        for t in range(seq_length):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))
            #outputs.append(h)
        outputs = torch.cat(outputs, dim=0).squeeze(1).transpose(0,1)
        backlog_out = self.fc(outputs)
        return backlog_out, (h, c)

    def new_instance(self):
        """
        Override this because our constructor accepts more args than the base class.
        And these things (dropout and activation) have to be available in the constructor.
        There must be a more elegant way to handle this.
        :return:
        """
        return self.__class__(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              learning_rate=self.learning_rate)

    def new_hidden_tensor(self, batch_size, device=None):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device), torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def get_extra_model_properties(self):
        extra_model_properties = {
        }
        return extra_model_properties

    def load_extra_model_properties(self, model_properties):
        return

