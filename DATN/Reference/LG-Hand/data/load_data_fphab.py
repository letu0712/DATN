"""
fuse training and testing

"""
import torch.utils.data as data

from data.common.camera import *
# from data.common.utils import deterministic_random
from data.common.generator import ChunkedGenerator
import numpy as np
import random
import torch






class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True):
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        # self.rescale = opt.rescale
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list, check = True,
                                                                                   subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_train, self.poses_train,
                                              self.poses_train_2d, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,out_all=opt.out_all)
            # print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list, check = False,
                                                                                subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_test, self.poses_test,
                                              self.poses_test_2d,
                                              pad=self.pad, augment=False)
            self.key_index = self.generator.saved_index
            # print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list):

        print('Loading 2D detections...')
        keypoints = np.load("data/data_2d_gt59.npz", allow_pickle=True)
        #keypoints = np.load(self.root_path + 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz',allow_pickle=True)
        keypoints = keypoints['positions_2d'].item()

        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    keypoints[subject][action][cam_idx] = kps
        return keypoints

    def fetch(self, dataset, subjects,check, subset=1, parse_3d_poses=True):
        """

        :param dataset:
        :param subjects:
        :param subset:
        :param parse_3d_poses:
        :return: for each pose dict it has key(subject,action,cam_index)
        """
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        # Train with sequence != 1,3
        if check == True:
            for subject in subjects:
                for action in self.keypoints[subject].keys():

                    if self.action_filter is not None:
                        found = False
                        for a in self.action_filter:
                            if action.startswith(a):
                                found = True
                                break
                        if not found:
                            continue

                    poses_2d = self.keypoints[subject][action]
                    for i in range(len(poses_2d)):  # Iterate across cameras
                        #if i != 0 and i != 2:
                        if i != 2:      #Train with seq != 3
                            out_poses_2d[(subject, action, i)] = poses_2d[i]

                    poses_3d = dataset[subject][action]
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): # Iterate across cameras
                        # if i != 0 and i != 2:
                        if i!= 2:       #Train with seq != 3
                            out_poses_3d[(subject, action, i)] = poses_3d[i]
                            out_camera_params[(subject, action, i)] = dataset.cameras()['intrinsic']

        # Test with sequence 1,3                    
        else:
            for subject in subjects:
                for action in self.keypoints[subject].keys():

                    if self.action_filter is not None:
                        found = False
                        for a in self.action_filter:
                            if action.startswith(a):
                                found = True
                                break
                        if not found:
                            continue                    

                    poses_2d = self.keypoints[subject][action]
                    for i in range(len(poses_2d)):  # Iterate across cameras
                        # if i == 0 or i == 2:
                        if i == 2:      #Only sequence 3 for test
                            out_poses_2d[(subject, action, i)] = poses_2d[i]


                    poses_3d = dataset[subject][action]
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): # Iterate across cameras
                        # if i == 0 or i == 2:
                        if i == 2:      #Only sequence 3 for test
                            out_poses_3d[(subject, action, i)] = poses_3d[i]
                            out_camera_params[(subject, action, i)] = dataset.cameras()['intrinsic']


        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None


        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        "Figure our how many sequences we have"

        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, reverse = self.generator.pairs[index]
        cam, gt_3D, input_2D, action, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, reverse)
        if self.train == False and self.test_aug:

            _, _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, reverse=reverse)
            input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float(1.0)

        return cam, gt_3D, input_2D_update, action, subject, scale, bb_box, cam_ind








