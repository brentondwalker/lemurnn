import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

from LatencyPredictor import LatencyPredictor


class BasicLink:
    """
    Basic Link
    """
    capacity = 0
    base_latency = 0
    queue_size = 0

    latency_predictor:LatencyPredictor = None


