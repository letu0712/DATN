import torch
import os 
import matplotlib.pyplot as plt
from data.common import eval_cal

pad = 1
if pad > 0:
    from nets.st_gcn_multi_frame import Model
else:
    from nets.st_gcn_single_frame import Model

from data.ho3d_dataset import HO3dDataset
import numpy as np
cam_intr = np.array([[614.627,   0.,    320.262],
 [  0.,    614.101, 238.469],
 [  0.,      0.,      1.,   ]])


connections = [(0,1,"red"), (1,6, "red"), (6,11, "red"), (11,16,"red"),
           (0,2,"yellow"), (2,7,"yellow"), (7,12,"yellow"), (12,17,"yellow"),
           (0,3,"green"), (3,8,"green"), (8,13,"green"), (13,18,"green"),
           (0,4, "blue"), (4,9, "blue"), (9,14, "blue"), (14,19, "blue"),
           (0,5,"purple"), (5,10,"purple"), (10,15,"purple"), (15,20,"purple")]

model_stgcn = Model().cuda()

model_stgcn.load_state_dict(torch.load("/home/lexuantu/DATN_HO3D/SST_GCN/checkpoint/checkpoint_epoch_26_14.3452.pth"))

test_data = HO3dDataset(phase="train", pad=pad)
sample2d, sample3d = test_data[56]


sample2d = sample2d.view(1, 3, 21, 2)

input_2D = sample2d.view(1, -1, 21, 2, 1).permute(0,3,1,2,4).type(torch.cuda.FloatTensor)
model_stgcn.eval()
output_3d = model_stgcn(input_2D, out_all_frame=True)
output_3d = output_3d.permute(0,2,3,4,1).contiguous().view(1, -1, 21, 3)



output_3d = output_3d.view(3,21,3).detach().cpu()
print("Error: ", eval_cal.mpjpe(output_3d, sample3d))
output_3d_single = output_3d.numpy()[1,:,:]

#output_3d_single = (np.linalg.inv(cam_intr).dot(output_3d_single.transpose())).transpose()



plt.clf()
plt.rcParams["figure.figsize"] = [10.0, 10.0]

ax = plt.axes(projection='3d')
x = output_3d_single[:21,0]
y = output_3d_single[:21,1]
z = output_3d_single[:21,2]

ax.scatter3D(x,y,z,c=z, cmap="gist_gray")

for conn in connections:
    x01 = output_3d_single[conn[0]:conn[1]+1:conn[1]-conn[0], 0]
    y01 = output_3d_single[conn[0]:conn[1]+1:conn[1]-conn[0], 1]
    z01 = output_3d_single[conn[0]:conn[1]+1:conn[1]-conn[0], 2]
    ax.plot(x01, y01, z01, color=conn[2])

plt.savefig("3d_pre.png")

#Visualize GT




sample3d = sample3d[1,:,:].numpy()
print(sample3d)
#sample3d = (np.linalg.inv(cam_intr).dot(sample3d.transpose())).transpose()
plt.clf()
plt.rcParams["figure.figsize"] = [10.0, 10.0]


ax = plt.axes(projection='3d')
x = sample3d[:21,0]
y = sample3d[:21,1]
z = sample3d[:21,2]

ax.scatter3D(x,y,z,c=z, cmap="gist_gray")

for conn in connections:
    x01 = sample3d[conn[0]:conn[1]+1:conn[1]-conn[0], 0]
    y01 = sample3d[conn[0]:conn[1]+1:conn[1]-conn[0], 1]
    z01 = sample3d[conn[0]:conn[1]+1:conn[1]-conn[0], 2]
    ax.plot(x01, y01, z01, color=conn[2])

plt.savefig("3d_gt.png")