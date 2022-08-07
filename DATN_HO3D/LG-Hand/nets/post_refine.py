import torch
import torch.nn as nn

from torch.autograd import Variable

import numpy as np
import random
import torch



import torch
import numpy as np 
import random


inter_channels = [128, 256]
fc_out = inter_channels[1]
fc_unit = 1024
class post_refine(nn.Module):


    def __init__(self):
        super().__init__()

        out_seqlen = 1
        in_channels = 2
        out_channels = 3
        n_joints = 21
        fc_in = out_channels*2*out_seqlen*n_joints    #3*2*1*21 = 126

        fc_out = in_channels * n_joints              #2*21=42
        self.post_refine = nn.Sequential(
            nn.Linear(fc_in, fc_unit),         #nn.Linear(126, 1024)
            nn.ReLU(),
            nn.Dropout(0.5,inplace=True),
            nn.Linear(fc_unit, fc_out),        #nn.Linear(1024, 42)
            nn.Sigmoid()
        )


    def forward(self, x, x_1):
        """

        :param x:  N*T*V*3
        :param x_1: N*T*V*2
        :return:
        """
        # data normalization
        N, T, V,_ = x.size()
        x_in = torch.cat((x, x_1), -1)  #N*T*V*5
        x_in = x_in.view(N, -1)         #shape=(N, T*V*5)



        score = self.post_refine(x_in).view(N,T,V,2)
        score_cm = Variable(torch.ones(score.size()), requires_grad=False).cuda() - score
        x_out = x.clone()
        x_out[:, :, :, :2] = score * x[:, :, :, :2] + score_cm * x_1[:, :, :, :2]

        return x_out