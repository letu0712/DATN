import os 
import numpy as np 

trainPath = "train"
listSeq = sorted(os.listdir(trainPath))

reorder_idx = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
change_pos = [0, 1, 5, 9,13,17,2, 6, 10,14,18,3,  7, 11,15,19, 4,  8, 12,16,20]
coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

points2d_train = []
points3d_train = []

points2d_val = []
points3d_val = []
for seq in listSeq:
    listMetas = sorted(os.listdir(os.path.join(trainPath, seq, "meta")))
    for metaFile in listMetas:
        pathMetaFile = os.path.join(trainPath,seq,"meta", metaFile)
        data = np.load(pathMetaFile, allow_pickle=True)
        cam_intr = data['camMat']
        print(cam_intr)
        hand3d = data['handJoints3D'][reorder_idx][change_pos]
        hand3d = cam_intr.dot(hand3d.transpose()).transpose()
        hand2d = (hand3d / hand3d[:, 2:])[:, :2]
        points2d_train.append(hand2d)
        points3d_train.append(hand3d)


np.save('points2d-train.npy', points2d_train)
np.save('points3d-train.npy', points3d_train)   #(66034, 21,3)


for seq in sorted(os.listdir("val")):
    listMetas = sorted(os.listdir(os.path.join("val", seq, "meta")))
    for metaFile in listMetas:
        pathMetaFile = os.path.join("val",seq,"meta", metaFile)
        data = np.load(pathMetaFile, allow_pickle=True)
        cam_intr = data['camMat']
        hand3d = data['handJoints3D'][reorder_idx][change_pos]
        hand3d = cam_intr.dot(hand3d.transpose()).transpose()
        hand2d = (hand3d / hand3d[:, 2:])[:, :2]
        points2d_val.append(hand2d)
        points3d_val.append(hand3d)


np.save('points2d-val.npy', points2d_val)
np.save('points3d-val.npy', points3d_val)   