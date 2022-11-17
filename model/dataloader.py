import json
import os
import random

import numpy as np

from torch.utils.data import Dataset, DataLoader


class AutomationDataset(Dataset):
    def __init__(
        self,
        data_dir,
        phase
    ):
        self.data_dir = data_dir
        self.phase = phase