import torch.nn as nn
from LinkEmuModel import LinkEmuModel


class NonManualRNN(LinkEmuModel):
    def __init__(self, input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, loadpath=None):
        super(NonManualRNN, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate, loadpath=loadpath)
        # after calling super(), the internal data fields will be populated.
        # either from the arguments passed in, or from loading a saved state
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          nonlinearity="relu", batch_first=True)

        # 1 for backlog + 2 or 1?? for dropped classification
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=3)


    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        combined_out = self.fc(out)

        # split backlog and dropped
        backlog_out = combined_out[:, :, 0:1]
        dropped_out = combined_out[:, :, 1:3]

        return backlog_out, dropped_out, hidden
