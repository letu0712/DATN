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

"""
    Case 1: sst-gcn + loss_angle + loss_direction    : 1780
    Case 2: sst-gcn                    :    1997
    Case 3: st-gcn      : 2025

"""

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

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

action = "squeeze_paper"

#open_soda_can
model = {}
opt = opts().parse()
opt.pad = 1


device = torch.device("cuda")

batch_cam = torch.zeros((1,4), dtype=torch.float32).cuda()
batch_cam[:,0] = 1395.7490
batch_cam[:,1] = 1395.7493
batch_cam[:,2] = 935.7325
batch_cam[:,3] = 540.6810

scale = torch.ones((1), dtype=torch.float32)

#============================Visualize GT=====================================

data_2D = np.load("/home/lexuantu/DATN/Reference/SST_GCN/data/data_2d_gt59.npz", allow_pickle=True)
data_2D = data_2D["positions_2d"].item()["Subject_3"][action][1]

data_3D = np.load("/home/lexuantu/DATN/Reference/SST_GCN/data/hand3d_gt_59.npz", allow_pickle=True)
data_3D = data_3D["positions_3d"].item()["Subject_3"][action][1]
print(data_3D.shape)        #(n_frames, 21, 3)

data_2D = torch.from_numpy(data_2D)
data_3D = torch.from_numpy(data_3D)

gt_folder = "visualize_gt/"
pre_folder_case1 = "visualize_pre/case1/"
pre_folder_case2 = "visualize_pre/case2/"
pre_folder_case3 = "visualize_pre/case3/"

os.system("rm -rf "+gt_folder)
os.system("rm -rf visualize_pre")

os.mkdir(gt_folder)
os.mkdir("visualize_pre")
os.mkdir(pre_folder_case1)
os.mkdir(pre_folder_case2)
os.mkdir(pre_folder_case3)


for i in range(data_3D.shape[0]):
    plt.clf()
    plt.rcParams["figure.figsize"] = [10.0, 10.0]
    ax = plt.axes(projection='3d')
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


#=================================Visualize Prediction case 1============================
model_st_gcn_case1 = Model(opt).cuda()

model_st_gcn_case1.load_state_dict(torch.load("/home/lexuantu/DATN/Reference/SST_GCN/results/3_frame/st_gcn/no_pose_refine/model_st_gcn_27_eva_xyz_1724.pth"))


model_st_gcn_case1.eval()


error_gt_case1 = []
error_angle_case1 = []
error_direction_case1 = []
for i in range(1, data_2D.size(0)):
    plt.clf()
    plt.rcParams["figure.figsize"] = [10.0, 10.0]
    
    ax = plt.axes(projection='3d')
    input_2D = torch.unsqueeze(data_2D[i-1:i+2,:,:], 0)
    gt_3D = torch.unsqueeze(data_3D[i-1:i+2,:,:],0)
    gt_3D = gt_3D.view(1, -1, 21, 3).type(torch.cuda.FloatTensor)
    gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)

    input_2D = input_2D.view(1,-1,21,2,1).permute(0,3,1,2,4).type(torch.cuda.FloatTensor)
    try:
        output_3D = model_st_gcn_case1(input_2D, out_all_frame=False)    #(1, 3, 1, 21, 1)
    except:
        break
    output_3D = output_3D.permute(0,2,3,4,1).contiguous().view(1, -1, 21, 3)    #(1,1,21,3)

    output_3D_single = output_3D


    output_3D_single = output_3D_single.view(21,3)
    target_3D = data_3D[i].view(21,3).cuda()

    error = round(mpjpe(output_3D_single, target_3D).detach().cpu().item(), 2)
    error_angle = round(eval_cal.cal_loss_angle_pred_gt(output_3D_single.view(1,1,21,3), target_3D.view(1,1,21,3)).detach().cpu().item(), 2)
    error_direction = round(eval_cal.cal_loss_direction(output_3D_single.view(1,1,21,3), target_3D.view(1,1,21,3)).detach().cpu().item(), 2)

    error_gt_case1.append(error)
    error_angle_case1.append(error_angle)
    error_direction_case1.append(error_direction)


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
    plt.title("LG-Hand: MPJPE("+ str(error) + ")\nAngle(" + str(error_angle) + ") - Direction(" + str(error_direction)+")")
    
    plt.savefig("visualize_pre/case1/ax"+str(i)+".png")



#=================================Visualize PRediction case2 ===================


model_st_gcn_case2 = Model(opt).cuda()

model_st_gcn_case2.load_state_dict(torch.load("/home/lexuantu/DATN/Reference/SST_GCN/results/3_frame/st_gcn/no_pose_refine/model_st_gcn_6_eva_xyz_1997.pth"))


model_st_gcn_case2.eval()


error_gt_case2 = []
error_angle_case2 = []
error_direction_case2 = []

for i in range(1, data_2D.size(0)):
    plt.clf()
    plt.rcParams["figure.figsize"] = [10.0, 10.0]
    ax = plt.axes(projection='3d')
    input_2D = torch.unsqueeze(data_2D[i-1:i+2,:,:], 0)
    gt_3D = torch.unsqueeze(data_3D[i-1:i+2,:,:],0)
    gt_3D = gt_3D.view(1, -1, 21, 3).type(torch.cuda.FloatTensor)
    gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)

    input_2D = input_2D.view(1,-1,21,2,1).permute(0,3,1,2,4).type(torch.cuda.FloatTensor)
    try:
        output_3D = model_st_gcn_case2(input_2D, out_all_frame=False)    #(1, 3, 1, 21, 1)
    except:
        break
    output_3D = output_3D.permute(0,2,3,4,1).contiguous().view(1, -1, 21, 3)    #(1,1,21,3)

    output_3D_single = output_3D

    output_3D_single = output_3D_single.view(21,3)
    target_3D = data_3D[i].view(21,3).cuda()
    error = round(mpjpe(output_3D_single, target_3D).detach().cpu().item(), 2)
    error_angle = round(eval_cal.cal_loss_angle_pred_gt(output_3D_single.view(1,1,21,3), target_3D.view(1,1,21,3)).detach().cpu().item(), 2)
    error_direction = round(eval_cal.cal_loss_direction(output_3D_single.view(1,1,21,3), target_3D.view(1,1,21,3)).detach().cpu().item(), 2)
    
    error_gt_case2.append(error)
    error_angle_case2.append(error_angle)
    error_direction_case2.append(error_direction)
    
    output_3D = output_3D_single.detach().cpu().numpy()
    
    x = output_3D[:,0]
    y = output_3D[:,1]
    z = output_3D[:,2]

    ax.scatter3D(x,y,z,c=z, cmap="gist_gray")
    for conn in connects:
        x01 = output_3D[conn[0]:conn[1]+1:conn[1]-conn[0], 0]
        y01 = output_3D[conn[0]:conn[1]+1:conn[1]-conn[0], 1]
        z01 = output_3D[conn[0]:conn[1]+1:conn[1]-conn[0], 2]
        ax.plot(x01, y01, z01, color=conn[2])
    plt.title("SST-GCN: MPJPE("+ str(error) + ")\nAngle(" + str(error_angle) + ") - Direction(" + str(error_direction)+")")
    plt.savefig("visualize_pre/case2/ax"+str(i)+".png")


#=================================Visualize Prediction case 3============================
model_st_gcn_case3 = Model(opt).cuda()

model_st_gcn_case3.load_state_dict(torch.load("/home/lexuantu/DATN/Reference/SST_GCN/results/3_frame/st_gcn/no_pose_refine/model_st_gcn_6_eva_xyz_2025.pth"))

model_st_gcn_case3.eval()

error_gt_case3 = []
error_angle_case3 = []
error_direction_case3 = []


for i in range(1, data_2D.size(0)):
    plt.clf()
    plt.rcParams["figure.figsize"] = [10.0, 10.0]
    
    ax = plt.axes(projection='3d')
    input_2D = torch.unsqueeze(data_2D[i-1:i+2,:,:], 0)
    gt_3D = torch.unsqueeze(data_3D[i-1:i+2,:,:],0)
    gt_3D = gt_3D.view(1, -1, 21, 3).type(torch.cuda.FloatTensor)
    gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)

    input_2D = input_2D.view(1,-1,21,2,1).permute(0,3,1,2,4).type(torch.cuda.FloatTensor)
    try:
        output_3D = model_st_gcn_case3(input_2D, out_all_frame=False)    #(1, 3, 1, 21, 1)
    except:
        break
    output_3D = output_3D.permute(0,2,3,4,1).contiguous().view(1, -1, 21, 3)    #(1,1,21,3)

    output_3D_single = output_3D



    output_3D_single = output_3D_single.view(21,3)
    target_3D = data_3D[i].view(21,3).cuda()
    error = round(mpjpe(output_3D_single, target_3D).detach().cpu().item(), 2)
    error_angle = round(eval_cal.cal_loss_angle_pred_gt(output_3D_single.view(1,1,21,3), target_3D.view(1,1,21,3)).detach().cpu().item(), 2)
    error_direction = round(eval_cal.cal_loss_direction(output_3D_single.view(1,1,21,3), target_3D.view(1,1,21,3)).detach().cpu().item(), 2)

    error_gt_case3.append(error)
    error_angle_case3.append(error_angle)
    error_direction_case3.append(error_direction)

    output_3D = output_3D_single.detach().cpu().numpy()
    
    x = output_3D[:,0]
    y = output_3D[:,1]
    z = output_3D[:,2]

    ax.scatter3D(x,y,z,c=z, cmap="gist_gray")
    for conn in connects:
        x01 = output_3D[conn[0]:conn[1]+1:conn[1]-conn[0], 0]
        y01 = output_3D[conn[0]:conn[1]+1:conn[1]-conn[0], 1]
        z01 = output_3D[conn[0]:conn[1]+1:conn[1]-conn[0], 2]
        ax.plot(x01, y01, z01, color=conn[2])
    plt.title("ST-GCN: MPJPE("+ str(error) + ")\nAngle(" + str(error_angle) + ") - Direction(" + str(error_direction)+")")
    
    plt.savefig("visualize_pre/case3/ax"+str(i)+".png")


# #===============================Export video====================================
import cv2
#export gt video



import numpy as np
mean_gt_case1 = np.round(np.mean(np.array(error_gt_case1)), 2)
mean_gt_case2 = np.round(np.mean(np.array(error_gt_case2)), 2)
mean_gt_case3 = np.round(np.mean(np.array(error_gt_case3)), 2)

mean_angle_case1 = np.round(np.mean(np.array(error_angle_case1)), 2)
mean_angle_case2 = np.round(np.mean(np.array(error_angle_case2)), 2)
mean_angle_case3 = np.round(np.mean(np.array(error_angle_case3)), 2)

mean_direction_case1 = np.round(np.mean(np.array(error_direction_case1)), 2)
mean_direction_case2 = np.round(np.mean(np.array(error_direction_case2)), 2)
mean_direction_case3 = np.round(np.mean(np.array(error_direction_case3)), 2)


frames_pre = os.listdir(pre_folder_case1)
img = cv2.imread(gt_folder+frames_pre[0])
size = (4000, 1000)
result = cv2.VideoWriter("Video_compare_("+action+").avi", cv2.VideoWriter_fourcc(*'MJPG'),5,size)
print(len(frames_pre))

plt.clf()
plt.figure(figsize=(30,10))
plt.plot([i for i in range(1, len(frames_pre)+1)], error_gt_case1, "b", label="LG-Hand", linewidth=3)
plt.plot([i for i in range(1, len(frames_pre)+1)], error_gt_case2, "r:", label="SST-GCN", linewidth=4)
plt.plot([i for i in range(1, len(frames_pre)+1)], error_gt_case3, "g--", label="ST-GCN", linewidth=3)
plt.xlabel("Frame", fontweight="bold",fontsize=40)
plt.ylabel("MPJPE (mm)", fontweight="bold",fontsize=40)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.legend(fontsize=36)
plt.savefig("mpjpe_video.png")

plt.clf()
plt.figure(figsize=(30,10))
plt.plot([i for i in range(1, len(frames_pre)+1)], error_angle_case1, "b", label="LG-Hand", linewidth=3)
plt.plot([i for i in range(1, len(frames_pre)+1)], error_angle_case2, "r:", label="SST-GCN", linewidth=4)
plt.plot([i for i in range(1, len(frames_pre)+1)], error_angle_case3, "g--", label="ST-GCN", linewidth=3)
plt.xlabel("Frame", fontweight="bold",fontsize=40)
plt.ylabel("Angle loss (radian)", fontweight="bold",fontsize=40)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.legend(fontsize=36)
plt.savefig("angle_video.png")


plt.clf()
plt.figure(figsize=(30,10))
plt.plot([i for i in range(1, len(frames_pre)+1)], error_direction_case1, "b", label="LG-Hand", linewidth=3)
plt.plot([i for i in range(1, len(frames_pre)+1)], error_direction_case2, "r:", label="SST-GCN", linewidth=4)
plt.plot([i for i in range(1, len(frames_pre)+1)], error_direction_case3, "g--", label="ST-GCN", linewidth=3)
plt.xlabel("Frame", fontweight="bold",fontsize=40)
plt.ylabel("Direction loss (radian)", fontweight="bold",fontsize=40)
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
plt.legend(fontsize=36)
plt.savefig("direction_video.png")



for i in range(1, len(frames_pre)+1):
    print(i)
    frame_gt = cv2.resize(cv2.imread(gt_folder + "ax"+str(i)+".png"), (1000,1000))
    frame_pre_case1 = cv2.resize(cv2.imread(pre_folder_case1 + "ax"+str(i)+".png"), (1000,1000))
    frame_pre_case2 = cv2.resize(cv2.imread(pre_folder_case2 + "ax"+str(i)+".png"), (1000,1000))
    frame_pre_case3 = cv2.resize(cv2.imread(pre_folder_case3 + "ax"+str(i)+".png"), (1000,1000))


    frame_compare = np.hstack((frame_gt, frame_pre_case1, frame_pre_case2, frame_pre_case3))
    # frame_compare = np.hstack((frame_compare, frame_pre_case2))

    result.write(frame_compare)

frame_result = np.ones((1000, 4000, 3), dtype=np.int32) * 255
cv2.putText(frame_result, "Case  GT  Angle  Direction", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2,  (0,0,0), 2)
cv2.putText(frame_result, "Case1 " +str(mean_gt_case1)+ " "+str(mean_angle_case1)+" " +str(mean_direction_case1), (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2,  (0,0,0), 2)
cv2.putText(frame_result, "Case2 " +str(mean_gt_case2)+ " "+str(mean_angle_case2)+" " +str(mean_direction_case2), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2,  (0,0,0), 2)
cv2.putText(frame_result, "Case3 " +str(mean_gt_case3)+ " "+str(mean_angle_case3)+" " +str(mean_direction_case3), (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 2,  (0,0,0), 2)

cv2.imwrite("Result.png", frame_result)

frame_result = cv2.imread("Result.png")
for i in range(10):
    result.write(frame_result)

result.release()





