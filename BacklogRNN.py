import torch
from torch import nn
from LinkEmuModel import LinkEmuModel

# # RNN Model Definition, L1loss, with 2 inputs(inter_pkt_times*link_capacity, pkt_sizes )

# Model definition
class BacklogRNN(LinkEmuModel):
    #input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, loadpath=None, nonlinearity='relu', dropout_rate=0.0
    def __init__(self, input_size=2, hidden_size=10, output_size=1, num_layers=1, learning_rate=0.001, loadpath=None, nonlinearity='relu', dropout_rate=0.0):
        super(BacklogRNN, self).__init__(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate, loadpath=loadpath)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)  # Linear layer for output
        return out, hidden

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
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def get_extra_model_properties(self):
        extra_model_properties = {
        }
        return extra_model_properties

    def load_extra_model_properties(self, model_properties):
        return
