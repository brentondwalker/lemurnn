import torch
import torch.nn as nn
from LinkEmuModel import LinkEmuModel


class DropLSTM(LinkEmuModel):
    def __init__(self, input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, loadpath=None, dropout_rate=0.0):
        self.model_name = "droplstm"
        self.nonlinearity = 'tanh'
        super(DropLSTM, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                       learning_rate=learning_rate, dropout_rate=dropout_rate, loadpath=loadpath)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        # 1 for backlog + 2 or 1?? for dropped classification
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=3)

    def forward(self, x, hidden=None):
        out, (hidden,cell) = self.lstm(x, hidden)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        combined_out = self.fc(out)

        # split backlog and dropped
        backlog_out = combined_out[:, :, 0:1]
        dropped_out = combined_out[:, :, 1:3]

        return backlog_out, dropped_out, (hidden, cell)

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
            'nonlinearity': self.nonlinearity
        }
        return extra_model_properties

    def load_extra_model_properties(self, model_properties):
        self.dropout_rate = model_properties.get('dropout_rate', 0.0)
        self.nonlinearity = model_properties.get('nonlinearity', 'relu')
        return

