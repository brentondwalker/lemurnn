import json
import os
import re

import torch
import torch.nn as nn
from torch import optim

class LinkEmuModel(nn.Module):

    def __init__(self, input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, loadpath=None):
        super(LinkEmuModel, self).__init__()
        #self.model_name = "none"
        self.input_size:int = input_size
        self.hidden_size:int = hidden_size
        self.num_layers:int = num_layers
        self.learning_rate:float = learning_rate
        if loadpath:
            print(f"loading model from {loadpath}")
            self.load_model_properties(loadpath)
        self.optimizer: optim.Optimizer = None
        self.training_directory = None
        self.seed:int = None

    def get_model_name(self):
        return self.model_name

    def set_optimizer(self, optimizer:optim.Optimizer=None, learning_rate=None):
        if not optimizer:
            if learning_rate:
                self.learning_rate = learning_rate
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            if learning_rate:
                print("WARNING: LinkEmuModel.set_optimizer(): optimizer was provided.  Ignoring learning_rate argument.")
        self.optimizer = optimizer

    def new_instance(self):
        return self.__class__(self.input_size, self.hidden_size, self.num_layers, self.learning_rate)

    def new_hidden_tensor(self, batch_size:int, device=None):
        return None

    def set_training_directory(self, training_directory):
        self.training_directory = training_directory

    def save_model_properties(self):
        model_properties_filename = f"{self.training_directory}/model-properties.json"
        model_properties = {'input_size': self.input_size,
                            #'output_size': self.output_size,
                            'hidden_size': self.hidden_size,
                            'num_layers': self.num_layers,
                            'learning_rate': self.learning_rate,
                            'seed': self.seed}
        # add any attributes that are specific to subclasses
        model_properties = dict(list(model_properties.items()) + list(self.get_extra_model_properties().items()))
        with open(model_properties_filename, "w") as model_properties_file:
            model_properties_file.write(json.dumps(model_properties))
            model_properties_file.write("\n")

    def get_extra_model_properties(self):
        extra_model_properties = {}
        return extra_model_properties

    def load_model_properties(self, loadpath):
        with open(f"{loadpath}/model-properties.json") as f:
            model_properties = json.load(f)
            self.input_size = model_properties['input_size']
            #self.output_size = model_properties['output_size']
            self.hidden_size = model_properties['hidden_size']
            self.num_layers = model_properties['num_layers']
            self.learning_rate = model_properties['learning_rate']
            self.seed = model_properties['seed']
            self.load_extra_model_properties(model_properties)
            print(f"\tloaded: input={self.input_size}\thidden={self.hidden_size}\tlayers={self.num_layers}")

    def load_extra_model_properties(self, model_properties):
        return

    def save_model_state(self, epoch):
        """
        To load it again:
        state = torch.load(filepath)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
        """
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }
        filepath = f"{self.training_directory}/modelstate-{epoch}.json"
        torch.save(state, filepath)

    def load_model_state(self, directory, device, epoch=-1):
        """
        Loads the specified model and optimizer state, and sets the class epoch.
        If you pass in epoch=-1, it will load the most recent model.
        kind-of inconsistent here.  The save function takes arguments, but the load
        function loads the values directly to the class and returns nothing.
        :param directory:
        :param epoch:
        :return:
        """
        if epoch < 0:
            pattern = re.compile(r'^modelstate-(\d+).json$')
            numbers = []
            try:
                for filename in os.listdir(directory):
                    match = pattern.match(filename)
                    if match:
                        numbers.append(int(match.group(1)))
            except FileNotFoundError:
                raise ValueError(f"Directory not found: {directory}")
            epoch = max(numbers)

        state_file_name = f"{directory}/modelstate-{epoch}.json"
        print(f"Loading state: {state_file_name}")
        state = torch.load(state_file_name, map_location=device)
        #self.model = NonManualRNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)
        self.load_state_dict(state['state_dict'])
        self.optimizer = optim.Adam(self.parameters())
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = epoch

