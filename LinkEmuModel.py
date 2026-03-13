import importlib
import inspect
import json
import os
import re

import torch
import torch.nn as nn
import wandb
from torch import optim, NoneType
from torch.export import Dim

MODEL_STRING_MAP = {
    "droprelurnn": "NonManualRNN",
    "droptanhrnn": "NonManualRNN",
    "droplstm": "DropLSTM",
    "droprelulstm": "DropReluLSTM",
    "dropgru": "DropGRU",
    "droprelurnn_ar": "NonManualRNNAR",
    "droptanhrnn_ar": "NonManualRNNAR",
    "droplstm_ar": "DropLSTMAR",
    "dropgru_ar": "DropGRUAR",
    "droprelurnn_ard": "NonManualRNNAR",
    "droptanhrnn_ard": "NonManualRNNAR",
    "droplstm_ard": "DropLSTMAR",
    "dropgru_ard": "DropGRUAR",
}

class LinkEmuModel(nn.Module):

    def __init__(self, input_size=4, hidden_size=2, num_layers=1, learning_rate=0.001, dropout_rate=0.0, loadpath=None):
        super(LinkEmuModel, self).__init__()
        #self.model_name = "none"
        self.input_size:int = input_size
        self.hidden_size:int = hidden_size
        self.num_layers:int = num_layers
        self.learning_rate:float = learning_rate
        self.dropout_rate:float = dropout_rate
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

    def save_model_state(self, epoch, wandb_run=None):
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
        if wandb_run:
            artifact = wandb.Artifact(name="best_model", type="model")
            artifact.add_file(filepath)
            wandb_run.log_artifact(artifact)
        self.export_torchscript(f"{self.training_directory}/modelstate-torchscript-{epoch}.pt")
        # skip ONNX for now.
        # too buggy and requires lots of special handling for LSTM
        #self.export_onnx(f"{self.training_directory}/modelstate-{epoch}.onnx")


    @classmethod
    def load_model_wandb(cls, run_name, device, epoch=-1, wandb_project='lemurnn'):
        """
        Fetches a W&B run's config to determine model architecture,
        instantiates the correct model subclass, and then loads the saved state.

        :param run_id: The 8-character W&B run ID (e.g., '1a2b3c4d')
        :param device: The torch device to load the tensors onto.
        :param epoch: The specific epoch to load, or -1 for the latest.
        :return: An instantiated model with loaded weights, ready for inference.
        """
        import os
        import wandb

        api = wandb.Api()

        # 1. Deduce Entity and Project to find the run
        if wandb.run is not None:
            # We are inside an active wandb.init() context
            entity = wandb.run.entity
            project = wandb.run.project
        else:
            # No active run. Deduce from environment and API defaults
            entity = os.environ.get("WANDB_ENTITY", api.default_entity)
            project = os.environ.get("WANDB_PROJECT")

        if not project:
            project = wandb_project

        # We only need the entity and project to search
        project_path = f"{entity}/{project}"
        print(f"Searching W&B project '{project_path}' for run: '{run_name}'...")

        # 1. Search for the run by its human-readable display name
        runs = api.runs(path=project_path, filters={"display_name": run_name})

        if len(runs) > 0:
            # We found it! Grab the first one in the list.
            run = runs[0]

            # (Optional) Warn if you accidentally have multiple runs with the exact same name
            if len(runs) > 1:
                print(f"Warning: Multiple runs found named '{run_name}'. Using the most recent one (ID: {run.id}).")
            else:
                print(f"Found run! (Internal ID: {run.id})")

        else:
            # 2. Fallback: Just in case you ACTUALLY passed the 8-character ID by mistake
            try:
                run = api.run(f"{project_path}/{run_name}")
                print(f"Found run directly by ID: {run.id}")
            except Exception:
                raise ValueError(
                    f"Could not find any run with display name or ID '{run_name}' in project '{project_path}'.")

        run_path = f"{entity}/{project}/{run.id}"
        print(f"Fetching config for run: {run_path}...")
        try:
            run = api.run(run_path)
        except Exception as e:
            raise ValueError(f"Could not find W&B run at '{run_path}'. Error: {e}")

        # 2. Extract required config parameters
        config = run.config

        autoregressive = config.get("autoregressive")
        hidden_size = config.get("hidden_size")
        learning_rate = config.get("learning_rate")
        model_type = config.get("model")
        nonlinearity = config.get("nonlinearity")
        num_layers = config.get("num_layers")
        use_deltas = config.get("use_deltas")

        print(f"Config loaded. Model type is '{model_type}'. Instantiating...")

        if model_type not in MODEL_STRING_MAP:
            raise ValueError(f"Model type '{model_type}' not found in MODEL_STRING_MAP.")

        class_name = MODEL_STRING_MAP[model_type]

        # dynamically import the correct model class
        try:
            # Assumes the file name matches the class name exactly
            model_module = importlib.import_module(class_name)
            ModelClass = getattr(model_module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to dynamically load '{class_name}'. Error: {e}")

        all_possible_args = {
            "autoregressive": autoregressive,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "nonlinearity": nonlinearity,
            "use_deltas": use_deltas
        }

        sig = inspect.signature(ModelClass)
        valid_params = sig.parameters.keys()

        filtered_kwargs = {k: v for k, v in all_possible_args.items() if k in valid_params and v is not None}
        print(f"Instantiating {ModelClass.__name__} with args: {list(filtered_kwargs.keys())}")

        # instantiate the model
        model = ModelClass(**filtered_kwargs)

        model.set_optimizer()
        print(f"MODEL NAME IS: {model.get_model_name()}")

        if model is None:
            raise NotImplementedError("Failed to instantiate a model")

        # Load the state into the newly instantiated model
        model.load_model_state_wandb(run.id, device, epoch=epoch, wandb_project=wandb_project)
        model.to(device)

        return model


    def load_model_state_wandb(self, run_id, device, epoch=-1, wandb_project='lemurnn'):
        """
        Loads the specified model and optimizer state from W&B.
        Automatically deduces the entity and project from the active run,
        environment variables, or W&B API defaults.

        :param run_id: The 8-character W&B run ID (e.g., '1a2b3c4d')
        :param device: The torch device to load the tensors onto.
        :param epoch: The specific epoch to load, or -1 for the latest.
        """
        import os
        import re
        import torch
        import wandb
        from torch import optim

        api = wandb.Api()

        # Deduce Entity and Project
        if wandb.run is not None:
            # We are inside an active wandb.init() context
            entity = wandb.run.entity
            project = wandb.run.project
        else:
            # No active run. Deduce from environment and API defaults
            entity = os.environ.get("WANDB_ENTITY", api.default_entity)
            project = os.environ.get("WANDB_PROJECT")

        if not project:
            project = wandb_project

        # Construct the full path
        run_path = f"{entity}/{project}/{run_id}"
        print(f"Connecting to W&B to fetch run: {run_path}...")

        try:
            # Fetch the run object from W&B
            run = api.run(run_path)
        except Exception as e:
            raise ValueError(f"Could not find W&B run at '{run_path}'. Error: {e}")

        # Get all logged artifacts for this run and filter for our 'best_model' lineage
        artifacts = run.logged_artifacts()
        model_artifacts = [a for a in artifacts if a.type == 'model' and 'best_model' in a.name]

        if not model_artifacts:
            raise FileNotFoundError(f"No 'best_model' artifacts found for run {run_path}")

        downloaded_dir = None
        target_file_name = None
        target_epoch = epoch

        if epoch < 0:
            # Find the artifact with the 'latest' alias
            latest_artifact = next((a for a in model_artifacts if 'latest' in a.aliases), model_artifacts[-1])
            print(f"Downloading latest artifact version ({latest_artifact.name})...")
            downloaded_dir = latest_artifact.download()

            # Find the highest epoch json file in this downloaded directory
            pattern = re.compile(r'^modelstate-(\d+).json$')
            numbers = []
            for filename in os.listdir(downloaded_dir):
                match = pattern.match(filename)
                if match:
                    numbers.append(int(match.group(1)))

            if not numbers:
                raise FileNotFoundError("No 'modelstate-{epoch}.json' files found in the latest artifact.")

            target_epoch = max(numbers)
            target_file_name = f"modelstate-{target_epoch}.json"

        else:
            # We are looking for a specific epoch
            target_file_name = f"modelstate-{epoch}.json"
            expected_artifact = None

            for a in model_artifacts:
                file_names = [f.name for f in a.files()]
                if target_file_name in file_names:
                    expected_artifact = a
                    break

            if not expected_artifact:
                raise FileNotFoundError(
                    f"Epoch {epoch} ({target_file_name}) not found in the logged artifacts for this run.")

            print(f"Downloading artifact version containing epoch {epoch}...")
            downloaded_dir = expected_artifact.download()

        # Construct the final path and load exactly as we do in load_model_state()
        state_file_path = os.path.join(downloaded_dir, target_file_name)
        print(f"Loading state: {state_file_path}")

        state = torch.load(state_file_path, map_location=device)
        self.load_state_dict(state['state_dict'])
        self.optimizer = optim.Adam(self.parameters())
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = target_epoch


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
        #XXX TODO: torch.jit.optimize_for_inference(torch.jit.script(mod.eval()))
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

