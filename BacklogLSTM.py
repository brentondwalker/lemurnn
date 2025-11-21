import torch
from torch import nn
from LinkEmuModel import LinkEmuModel


class BacklogLSTM(LinkEmuModel):

    def __init__(self, input_size=2, hidden_size=10, output_size=1, num_layers=1, learning_rate=0.001, loadpath=None):
        super(BacklogLSTM, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate, loadpath=loadpath)
        self.model_name = "backloglstm"
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        x, (h,c) = self.lstm(x)
        # take only the last output - why??!
        #if not full_trace:
        #    x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x, (h,c)

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
