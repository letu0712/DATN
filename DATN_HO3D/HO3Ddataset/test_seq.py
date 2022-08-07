import os
import pickle as pkl 
import numpy as np 
import matplotlib.pyplot as plt

connections = [(0,1,"red"), (1,6, "red"), (6,11, "red"), (11,16,"red"),
           (0,2,"yellow"), (2,7,"yellow"), (7,12,"yellow"), (12,17,"yellow"),
           (0,3,"green"), (3,8,"green"), (8,13,"green"), (13,18,"green"),
           (0,4, "blue"), (4,9, "blue"), (9,14, "blue"), (14,19, "blue"),
           (0,5,"purple"), (5,10,"purple"), (10,15,"purple"), (15,20,"purple")]

points3d = np.load('points3d-train.npy', allow_pickle=True)
print(points3d.shape)

for i in range(points3d.shape[0]):
    hand3d = points3d[i,:,:]
    plt.clf()
    plt.rcParams["figure.figsize"] = [10.0, 10.0]


    ax = plt.axes(projection='3d')
    ax.set_xlim(-0.1,0.1)
    ax.set_ylim(-0.1,0.1)
    ax.set_zlim(-1,1)
    x = hand3d[:,0]
    y = hand3d[:,1]
    z = hand3d[:,2]

    ax.scatter3D(x,y,z,c=z, cmap="gist_gray")

    for conn in connections:
        x01 = hand3d[conn[0]:conn[1]+1:conn[1]-conn[0], 0]
        y01 = hand3d[conn[0]:conn[1]+1:conn[1]-conn[0], 1]
        z01 = hand3d[conn[0]:conn[1]+1:conn[1]-conn[0], 2]
        ax.plot(x01, y01, z01, color=conn[2])

    plt.savefig("3d_gt_sample1.png")


