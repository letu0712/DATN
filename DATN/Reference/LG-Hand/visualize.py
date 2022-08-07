import torch
import torch.nn as nn
import numpy as np
from nets.st_gcn_multi_frame import Model
from nets.post_refine import post_refine
from opt1 import opts
import matplotlib.pyplot as plt
import os 
from utils.utils1 import get_uvd2xyz
import data.common.eval_cal as eval_cal 
import cv2


connects = [(0,1,"red"), (1,6, "red"), (6,11, "red"), (11,16,"red"),
           (0,2,"yellow"), (2,7,"yellow"), (7,12,"yellow"), (12,17,"yellow"),
           (0,3,"green"), (3,8,"green"), (8,13,"green"), (13,18,"green"),
           (0,4, "blue"), (4,9, "blue"), (9,14, "blue"), (14,19, "blue"),
           (0,5,"purple"), (5,10,"purple"), (10,15,"purple"), (15,20,"purple")]

actions = ["open_milk", "open_soda_can", "scratch_sponge", "give_coin", "wash_sponge", "open_letter",
   "unfold_glasses", "open_peanut_butter", "read_letter", "pour_liquid_soap", "pour_wine", "tear_paper", "open_wallet", "put_tea_bag", "high_five", 
  "charge_cell_phone", "handshake", "light_candle", "toast_wine", "scoop_spoon", "flip_sponge", 
  "receive_coin", "close_peanut_butter", "use_flash", "flip_pages", "close_liquid_soap", "close_milk", 
  'squeeze_sponge', 'pour_juice_bottle', 'pour_milk', 'take_letter_from_enveloppe', 'use_calculator', 
  "write", "put_salt", "clean_glasses", "prick", "open_liquid_soap", "open_juice_bottle", 
  "close_juice_bottle", "sprinkle", "give_card", "drink_mug", "stir", "put_sugar", "squeeze_paper"]

action = "pour_wine"

device = torch.device("cuda")

batch_cam = torch.zeros((1,4), dtype=torch.float32).cuda()
batch_cam[:,0] = 1395.7490
batch_cam[:,1] = 1395.7493
batch_cam[:,2] = 935.7325
batch_cam[:,3] = 540.6810

scale = torch.ones((1), dtype=torch.float32)

opt = opts().parse()
opt.pad = 1

#=====================Visualize GT======================
data_2D = np.load("/home/lexuantu/DATN/Reference/SST_GCN/data/data_2d_gt59.npz", allow_pickle=True)
data_2D = data_2D["positions_2d"].item()["Subject_3"][action][1]

data_3D = np.load("/home/lexuantu/DATN/Reference/SST_GCN/data/hand3d_gt_59.npz", allow_pickle=True)
data_3D = data_3D["positions_3d"].item()["Subject_3"][action][1]

data_2D = torch.from_numpy(data_2D)
data_3D = torch.from_numpy(data_3D)


gt_folder = "visualize_gt/"
os.system("rm -rf "+gt_folder)
os.mkdir(gt_folder)
for i in range(data_3D.shape[0]):
    plt.clf()
    plt.rcParams["figure.figsize"] = [10.0, 10.0]
    ax = plt.axes(projection='3d')
    ax.set_xlim(-80,80)
    ax.set_ylim(0,150)
    ax.set_zlim(200,500)
    x = data_3D[i][:,0]
    y = data_3D[i][:,1]
    z = data_3D[i][:,2]

    ax.scatter3D(x,y,z,c=z, cmap="gist_gray")
    for conn in connects:
        x01 = data_3D[i][conn[0]:conn[1]+1:conn[1]-conn[0], 0]
        y01 = data_3D[i][conn[0]:conn[1]+1:conn[1]-conn[0], 1]
        z01 = data_3D[i][conn[0]:conn[1]+1:conn[1]-conn[0], 2]
        ax.plot(x01, y01, z01, color=conn[2])
    plt.title("Ground truth "+str(i)+" "+ action)
    plt.savefig("visualize_gt/ax"+str(i)+".png")

#===============================Visualize Pred==========================
models_dict = {
    "LG-Hand" : "/home/lexuantu/DATN/Reference/SST_GCN/results/3_frame/st_gcn/no_pose_refine/model_st_gcn_27_eva_xyz_1724.pth",
    "SST-GCN" : "/home/lexuantu/DATN/Reference/SST_GCN/results/3_frame/st_gcn/no_pose_refine/model_st_gcn_6_eva_xyz_1997.pth",
    "ST-GCN" : "/home/lexuantu/DATN/Reference/SST_GCN/results/3_frame/st_gcn/no_pose_refine/model_st_gcn_6_eva_xyz_2025.pth"
}


for key, value in models_dict.items():
    model_stgcn = Model(opt).cuda()
    model_stgcn.load_state_dict(torch.load(models_dict[key]))
    model_stgcn.eval()
    os.system("rm -rf visualize_pre/"+ key)
    os.mkdir("visualize_pre/"+key)
    for i in range(1, data_2D.size(0)):

        input_2D = torch.unsqueeze(data_2D[i-1:i+2,:,:], 0)
        gt_3D = torch.unsqueeze(data_3D[i-1:i+2,:,:],0).view(1, -1, 21, 3).type(torch.cuda.FloatTensor)
        gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)

        input_2D = input_2D.view(1,-1,21,2,1).permute(0,3,1,2,4).type(torch.cuda.FloatTensor)
        try:
            output_3D = model_stgcn(input_2D, out_all_frame=False)    #(1, 3, 1, 21, 1)
        except:
            break
        output_3D = output_3D.permute(0,2,3,4,1).contiguous().view(1, -1, 21, 3)    #(1,1,21,3)

        output_3D_single = output_3D
        output_3D_single = output_3D_single.view(21,3)
        target_3D = data_3D[i].view(21,3).cuda()

        error = round(eval_cal.mpjpe(output_3D_single, target_3D).detach().cpu().item(), 2)
        error_angle = round(eval_cal.cal_loss_angle_pred_gt(output_3D_single.view(1,1,21,3), target_3D.view(1,1,21,3)).detach().cpu().item(), 2)
        error_direction = round(eval_cal.cal_loss_direction(output_3D_single.view(1,1,21,3), target_3D.view(1,1,21,3)).detach().cpu().item(), 2)

        plt.clf()
        plt.rcParams["figure.figsize"] = [10.0, 10.0]
        ax = plt.axes(projection='3d')
        ax.set_xlim(-80,80)
        ax.set_ylim(0,150)
        ax.set_zlim(200,500)
        output_3D = output_3D_single.view(21,3).detach().cpu().numpy()

        x = output_3D[:,0]
        y = output_3D[:,1]
        z = output_3D[:,2]

        ax.scatter3D(x,y,z,c=z, cmap="gist_gray")
        for conn in connects:
            x01 = output_3D[conn[0]:conn[1]+1:conn[1]-conn[0], 0]
            y01 = output_3D[conn[0]:conn[1]+1:conn[1]-conn[0], 1]
            z01 = output_3D[conn[0]:conn[1]+1:conn[1]-conn[0], 2]
            ax.plot(x01, y01, z01, color=conn[2])
        plt.title(str(key)+": MPJPE("+ str(error) + ")\nAngle(" + str(error_angle) + ") - Direction(" + str(error_direction)+")")

        print("visualize_pre/"+str(key)+"/ax"+str(i)+".png")
        plt.savefig("visualize_pre/"+str(key)+"/ax"+str(i)+".png")

num_frames = data_2D.size(0) - 2
size = (3000, 2000)
result = cv2.VideoWriter("Video_compare_("+action+").avi", cv2.VideoWriter_fourcc(*'MJPG'),10,size)

for i in range(1, num_frames+1):
    frame_gt = cv2.resize(cv2.imread(gt_folder + "ax"+str(i)+".png"), (1500,1000))
    frame_LGHand = cv2.resize(cv2.imread("visualize_pre/LG-Hand/" + "ax"+str(i)+".png"), (1500,1000))
    frame_SSTGCN = cv2.resize(cv2.imread("visualize_pre/SST-GCN/" + "ax"+str(i)+".png"), (1500,1000))
    frame_STGCN = cv2.resize(cv2.imread("visualize_pre/ST-GCN/" + "ax"+str(i)+".png"), (1500,1000))

    frame_compare_row1 = np.hstack((frame_gt, frame_LGHand))
    frame_compare_row2 = np.hstack((frame_SSTGCN, frame_STGCN))
    frame_compare = np.vstack((frame_compare_row1, frame_compare_row2))
    result.write(frame_compare)

result.release()