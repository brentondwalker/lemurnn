import json
import os
import re

import torch
import torch.nn as nn
from torch import optim
from torch.export import Dim


class LinkEmuModel(nn.Module):

    def __init__(self, input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, dropout_rate=0.0, loadpath=None):
        super(LinkEmuModel, self).__init__()
        #self.model_name = "none"
        self.input_size:int = input_size
        self.hidden_size:int = hidden_size
        self.num_layers:int = num_layers
        self.learning_rate:float = learning_rate
        self.dropout_rate = dropout_rate
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

    def get_model_param_string(self):
        extra_stuff = ""
        if self.dropout_rate > 0.0:
            extra_stuff += f"_dr{self.dropout_rate}"
        return f"l{self.num_layers}_h{self.hidden_size}{extra_stuff}"

    def new_instance(self):
        return self.__class__(self.input_size, self.hidden_size, self.num_layers, self.learning_rate)

    def new_hidden_tensor(self, batch_size:int, device=None):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

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
        self.export_torchscript(f"{self.training_directory}/modelstate-torchscript-{epoch}.pt")
        # skip ONNX for now.
        # too buggy and requires lots of special handling for LSTM
        #self.export_onnx(f"{self.training_directory}/modelstate-{epoch}.onnx")

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


    def export_torchscript(self, filename, state_dict=None):
        """
        :param filename:
        :param state_dict:
        :return:
        """
        BATCH_SIZE = 1
        SEQ_LEN = 1

        model = self.new_instance()
        if state_dict:
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(self.state_dict())
        model.eval()
        dummy_x = torch.randn(BATCH_SIZE, SEQ_LEN, model.input_size)
        dummy_hidden = model.new_hidden_tensor(batch_size=BATCH_SIZE)

        traced_model = torch.jit.trace(model, (dummy_x, dummy_hidden))
        traced_model.save(filename)


    def export_onnx(self, filename, state_dict=None):
        """
        Actual useful info:
        https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html

        :param filename:
        :param state_dict:
        :return:
        """
        # Model parameters (using defaults from your __init__)
        INPUT_SIZE = self.input_size

        model = self.new_instance()
        if state_dict:
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(self.state_dict())
        model.eval()

        BATCH_SIZE = 1
        SEQ_LEN = 1

        # Create dummy inputs that match the model's forward() signature: (x, hidden)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)
        hidden = self.new_hidden_tensor(BATCH_SIZE)
        dummy_inputs = (x, hidden)

        # Define file path and I/O names
        onnx_file_path = filename
        input_names = ["x", "hidden"]
        output_names = ["backlog_out", "dropped_out", "output_hidden"]

        print(f"Exporting model to {onnx_file_path} (Opset Version 18)...")

        # Using RNNs with single-step inference (SEQ_LEN=1) or NUM_LAYERS=1 with onnx causes all sorts
        # of problems.  The only solution seems to be to make those dimensions dynamic.
        # This is not a big deal, but I wish it were documented better, so I would not have spent a whole
        # day fighting with it.
        dynamic_shapes = {
            "x": (Dim.AUTO, Dim.AUTO, Dim.STATIC),
            "hidden": (Dim.AUTO, Dim.STATIC, Dim.STATIC),
        }

        try:
            torch.onnx.export(
                model,
                dummy_inputs,
                onnx_file_path,
                opset_version=18,
                input_names=input_names,
                output_names=output_names,
                dynamic_shapes=dynamic_shapes
            )
            print("\nExport successful!")
            print("This model should now accept inputs with any sequence_length (e.g., 1).")

        except Exception as e:
            print(f"\nExport failed: {e}")

