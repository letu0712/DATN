import numpy as np
import random
import torch

import torch
import numpy as np 
import random

class MocapDataset:
    def __init__(self, fps, skeleton):
        self._skeleton = skeleton
        self._fps = fps
        self._data = None  # Must be filled by subclass
        self._cameras = None  # Must be filled by subclass


    def __getitem__(self, key):
        return self._data[key]

    def subjects(self):
        return self._data.keys()

    def fps(self):
        return self._fps

    def skeleton(self):
        return self._skeleton

    def cameras(self):
        return self._cameras

    def supports_semi_supervised(self):
        # This method can be overridden
        return False