import numpy as np

points2d = np.load("points2d-train.npy", allow_pickle=True)
points3d = np.load("points3d-train.npy", allow_pickle=True)

print(points2d[0])
print(points3d[0])