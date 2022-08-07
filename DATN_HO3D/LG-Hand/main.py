import torch
import numpy as np
import random
from data.ho3d_dataset import HO3dDataset
import data.common.eval_cal as eval_cal 
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
from utils.utils1 import *

seed = 1

learning_rate = 1e-4
pro_train = True 
pro_test = False
workers = 6
optimizer = "SGD"
weight_decay = 0.0001
nepoch = 30
batch_size = 256

#Reload model
stgcn_reload = False

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

pad = 1
if pad > 0:
    from nets.st_gcn_multi_frame import Model
else:
    from nets.st_gcn_single_frame import Model

model_stgcn = Model().cuda()


train_data = HO3dDataset(phase="train", pad=pad)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                shuffle=True)


test_data = HO3dDataset(phase="val", pad=pad)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                shuffle=False)


#SEt optimizer
all_param = []
all_param += list(model_stgcn.parameters())
if optimizer == 'SGD':
    optimizer_all = optim.SGD(all_param, lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay)
elif optimizer == 'Adam':
    optimizer_all = optim.Adam(all_param, lr=learning_rate, amsgrad=True)
optimizer_all_scheduler = optim.lr_scheduler.StepLR(optimizer_all, step_size=5, gamma=0.1)

#5.Set criterion
criterion = {}
criterion['MSE'] = nn.MSELoss(reduction="mean").cuda()
criterion['L1'] = nn.L1Loss(reduction="mean").cuda()

n_joints = 21
out_channels = 3
in_channels = 2

loader_dict = {
    "train": train_dataloader,
    "val": test_dataloader
}

best_model = float('inf')       #Loss of best model

for epoch in range(0, nepoch):
    print('======>>>>> Online epoch: #%d <<<<<======' % (epoch))
    torch.cuda.synchronize()
    

    for phase in ["train", "val"]:
        loss_all_sum = {
                'loss_thumb':AccumLoss(), 
                'loss_index':AccumLoss(), 
                'loss_middle':AccumLoss(), 
                'loss_ring':AccumLoss(), 
                'loss_pinky':AccumLoss(), 
                'loss_Wrist':AccumLoss(), 
                'loss_MCP':AccumLoss(), 
                'loss_PIP':AccumLoss(), 
                'loss_DIP':AccumLoss(), 
                'loss_TIP':AccumLoss()}

        if phase == "train":
            model_stgcn.train()
        else: 
            model_stgcn.eval()
            
        epoch_loss_sum = 0
        epoch_loss_gt = 0
        epoch_loss_angle = 0
        epoch_loss_dist = 0
        epoch_loss_direction = 0

        num_all_data = 0                        #13234
        num_correct = [0 for i in range(0,250)]
        thresholds = [i for i in range(0,250)]
        for i, data in enumerate(tqdm(loader_dict[phase])):
            input_2D, gt_3D = data
            input_2D = input_2D.cuda()
            gt_3D = gt_3D.cuda()
            N = input_2D.size(0)
            num_all_data += N
            out_target = gt_3D.clone().view(N, -1, n_joints, out_channels)
            gt_3D = gt_3D.view(N, -1, n_joints, out_channels).type(torch.cuda.FloatTensor)

            input_2D = input_2D.view(N, -1, n_joints, in_channels, 1).permute(0,3,1,2,4).type(torch.cuda.FloatTensor)
            output_3D = model_stgcn(input_2D, out_all_frame=True)
            output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, n_joints, out_channels)

            loss_gt = eval_cal.mpjpe(output_3D, out_target)

            output_3D_single = output_3D[:,1,:,:]
            
            gt_3D_single = out_target[:,1,:,:]
            #Number correct data (> threshold)
            if phase =="val" and epoch==26:
                for i in range(N):
                    for threshold in thresholds:
                        if torch.norm(output_3D_single[i,:,:] - gt_3D_single[i,:,:]).item() < threshold:
                            num_correct[threshold] += 1

            loss_dist = eval_cal.cal_distant(output_3D, out_target)
            loss_direction = eval_cal.cal_loss_direction(output_3D, out_target)
            loss_angle = eval_cal.cal_loss_angle_pred_gt(output_3D, out_target)

            loss = loss_gt + 0.1*loss_dist + 0.1*loss_angle + 0.01*loss_direction

            epoch_loss_sum = epoch_loss_sum + loss.detach().cpu().item() * N
            epoch_loss_gt = epoch_loss_gt + loss_gt.detach().cpu().item() * N
            epoch_loss_angle = epoch_loss_angle + loss_angle.detach().cpu().item() * N
            epoch_loss_dist = epoch_loss_dist + loss_dist.detach().cpu().item() * N
            epoch_loss_direction = epoch_loss_direction + loss_direction.detach().cpu().item() * N
            
            Thumb_target = out_target[:,:,1::5]    #1::5 = 1,6,11,16  (start at 1 with step = 5)
            Thumb_pre = output_3D[:,:,1::5]

            Index_target = out_target[:,:,2::5]
            Index_pre = output_3D[:,:,2::5]

            Middle_target = out_target[:,:,3::5]
            Middle_pre = output_3D[:,:,3::5]

            Ring_target = out_target[:,:,4::5]
            Ring_pre = output_3D[:,:,4::5]

            Pinky_target = out_target[:,:,5::5]
            Pinky_pre = output_3D[:,:,5::5]
            
            
            # Moi khop cua ban tay
            Wrist_target = out_target[:,:,:1]
            Wrist_pre = output_3D[:,:,:1]

            MCP_target = out_target[:,:,1:6]
            MCP_pre = output_3D[:,:,1:6]

            PIP_target = out_target[:,:,6:11]
            PIP_pre = output_3D[:,:,6:11]

            DIP_target = out_target[:,:,11:16]
            DIP_pre = output_3D[:,:,11:16]

            TIP_target = out_target[:,:,16:21]
            TIP_pre = output_3D[:,:,16:21]

            loss_thumb = eval_cal.mpjpe(Thumb_target, Thumb_pre)
            loss_index = eval_cal.mpjpe(Index_target, Index_pre)
            loss_middle = eval_cal.mpjpe(Middle_target, Middle_pre)
            loss_ring = eval_cal.mpjpe(Ring_target, Ring_pre)
            loss_pinky = eval_cal.mpjpe(Pinky_target, Pinky_pre)
            
            loss_Wrist = eval_cal.mpjpe(Wrist_target, Wrist_pre)
            loss_MCP = eval_cal.mpjpe(MCP_target, MCP_pre)
            loss_PIP = eval_cal.mpjpe(PIP_target, PIP_pre)
            loss_DIP = eval_cal.mpjpe(DIP_target, DIP_pre)
            loss_TIP = eval_cal.mpjpe(TIP_target, TIP_pre)  
            
            loss_all_sum['loss_thumb'].update(loss_thumb.detach().cpu().numpy() * N, N)
            loss_all_sum['loss_index'].update(loss_index.detach().cpu().numpy() * N, N)
            loss_all_sum['loss_middle'].update(loss_middle.detach().cpu().numpy() * N, N)
            loss_all_sum['loss_ring'].update(loss_ring.detach().cpu().numpy() * N, N)
            loss_all_sum['loss_pinky'].update(loss_pinky.detach().cpu().numpy() * N, N)
        
            loss_all_sum['loss_Wrist'].update(loss_Wrist.detach().cpu().numpy() * N, N)
            loss_all_sum['loss_MCP'].update(loss_MCP.detach().cpu().numpy() * N, N)
            loss_all_sum['loss_PIP'].update(loss_PIP.detach().cpu().numpy() * N, N)
            loss_all_sum['loss_DIP'].update(loss_DIP.detach().cpu().numpy() * N, N)
            loss_all_sum['loss_TIP'].update(loss_TIP.detach().cpu().numpy() * N, N)
            
            if phase == "train":
                optimizer_all.zero_grad()
                loss.backward()
                optimizer_all.step()
        print("num_all_data: ", num_all_data)
        epoch_loss_sum = epoch_loss_sum / num_all_data
        epoch_mpjpe = epoch_loss_gt / num_all_data
        epoch_dist = epoch_loss_dist / num_all_data
        epoch_angle = epoch_loss_angle / num_all_data
        epoch_direction = epoch_loss_direction / num_all_data
        if phase == "val":
            print("Epoch_loss_sum: ", epoch_loss_sum)
            print("MPJPE: ", epoch_mpjpe)
            print("Dist_loss: ", epoch_dist)
            print("Angle_loss: ", epoch_angle)
            print("Direction loss: ", epoch_direction)



            print('loss thumb each frame of 1 sample: %f mm' % (loss_all_sum['loss_thumb'].avg))
            print('loss index each frame of 1 sample: %f mm' % (loss_all_sum['loss_index'].avg))
            print('loss middle each frame of 1 sample: %f mm' % (loss_all_sum['loss_middle'].avg))
            print('loss ring each frame of 1 sample: %f mm' % (loss_all_sum['loss_ring'].avg))
            print('loss pinky each frame of 1 sample: %f mm' % (loss_all_sum['loss_pinky'].avg))  
            print('\n')
            print('loss Wrist each frame of 1 sample: %f mm' % (loss_all_sum['loss_Wrist'].avg))
            print('loss MCP each frame of 1 sample: %f mm' % (loss_all_sum['loss_MCP'].avg))
            print('loss PIP each frame of 1 sample: %f mm' % (loss_all_sum['loss_PIP'].avg))
            print('loss DIP each frame of 1 sample: %f mm' % (loss_all_sum['loss_DIP'].avg))
            print('loss TIP each frame of 1 sample: %f mm' % (loss_all_sum['loss_TIP'].avg))              
            if epoch_mpjpe < best_model:
                best_model = epoch_mpjpe
                os.system("rm -rf checkpoint")
                os.mkdir("checkpoint")
                torch.save(model_stgcn.state_dict(), "checkpoint/checkpoint_epoch_"+str(epoch)+"_"+str(round(epoch_mpjpe, 4))+".pth")         

        print("Num correct: ", num_correct)
        
        print(phase + " loss ground truth: ", epoch_mpjpe)
        


