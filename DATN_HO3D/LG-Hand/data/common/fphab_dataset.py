"""
refined from https://github.com/facebookresearch/VideoPose3D
"""
import numpy as np
import copy
from data.common.skeleton import Skeleton
from data.common.mocap_dataset import MocapDataset
from data.common.camera import normalize_screen_coordinates

import numpy as np
import random
import torch


fphab_skeleton = Skeleton(parents=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15,16,17,18,19,20])

fphab_cameras_extrinsic_params = {
    'center': [935.732544, 540.681030],
    'focal_length': [1395.749023, 1395.749268],
}

class HandfphabDataset(MocapDataset):
    def __init__(self, path, opt,remove_static_joints=True):
        super().__init__(fps=50, skeleton=fphab_skeleton)
        self.train_list = ['Subject_1', 'Subject_2', 'Subject_3', 'Subject_4','Subject_5', 'Subject_6']
        self.test_list = ['Subject_1', 'Subject_2', 'Subject_3', 'Subject_4','Subject_5', 'Subject_6']
        self._cameras = copy.deepcopy(fphab_cameras_extrinsic_params)

        for k, v in self._cameras.items():
            self._cameras[k] = np.array(v, dtype='float32')

        # Add intrinsic parameters vector
        self._cameras['intrinsic'] = np.concatenate((self._cameras['focal_length'],self._cameras['center']))  

        # Load serialized dataset
        data = np.load(path,allow_pickle=True)['positions_3d'].item()

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, camm in actions.items():
                self._data[subject][action_name] = []
                for i in range(len(camm)):
                    self._data[subject][action_name].append(camm[i])
                


    def supports_semi_supervised(self):
        return True
