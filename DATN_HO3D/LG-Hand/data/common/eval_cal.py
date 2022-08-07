import torch
import numpy as np
import torch.nn as nn

import numpy as np
import random
import torch

import torch
import numpy as np 
import random
seed = 1

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def cal_distant(predicted, target):
    assert predicted.shape == target.shape     #shape = (256,3,21,3)

    pre_1 = torch.zeros(predicted.size()[0:-2]).cuda()   #shape = (256, 3)
    pre_2 = torch.zeros(predicted.size()[0:-2]).cuda()
    pre_3 = torch.zeros(predicted.size()[0:-2]).cuda()
    pre_4 = torch.zeros(predicted.size()[0:-2]).cuda()
    pre_5 = torch.zeros(predicted.size()[0:-2]).cuda()
    for i in range(1,16,5):
        pre_1 +=  torch.norm(predicted[:,:,i] - predicted[:,:,i+5], dim=len(target.shape) - 2)      #sum of length connected by joints: 1,6,11... shape=(256,3)
        pre_2 +=  torch.norm(predicted[:,:,i+1] - predicted[:,:,i+6], dim=len(target.shape) - 2)    #joint: 2,7,12
        pre_3 +=  torch.norm(predicted[:,:,i+2] - predicted[:,:,i+7], dim=len(target.shape) - 2)    #joint: 3,8,13
        pre_4 +=  torch.norm(predicted[:,:,i+3] - predicted[:,:,i+8], dim=len(target.shape) - 2)    #joint: 4,9,14
        pre_5 +=  torch.norm(predicted[:,:,i+4] - predicted[:,:,i+9], dim=len(target.shape) - 2)    #joint: 5,10, 15

    out_1 = torch.zeros(target.size()[0:-2]).cuda()
    out_2 = torch.zeros(target.size()[0:-2]).cuda()
    out_3 = torch.zeros(target.size()[0:-2]).cuda()
    out_4 = torch.zeros(target.size()[0:-2]).cuda()
    out_5 = torch.zeros(target.size()[0:-2]).cuda()
    for i in range(1,16,5):
        out_1 +=  torch.norm(target[:,:,i] - target[:,:,i+5], dim=len(target.shape) - 2)
        out_2 +=  torch.norm(target[:,:,i+1] - target[:,:,i+6], dim=len(target.shape) - 2)
        out_3 +=  torch.norm(target[:,:,i+2] - target[:,:,i+7], dim=len(target.shape) - 2)
        out_4 +=  torch.norm(target[:,:,i+3] - target[:,:,i+8], dim=len(target.shape) - 2)
        out_5 +=  torch.norm(target[:,:,i+4] - target[:,:,i+9], dim=len(target.shape) - 2)

    criterion = nn.L1Loss(reduction="mean")
    l1 = torch.mean(criterion(pre_1, out_1))
    l2 = torch.mean(criterion(pre_2, out_2))
    l3 = torch.mean(criterion(pre_3, out_3))
    l4 = torch.mean(criterion(pre_4, out_4))
    l5 = torch.mean(criterion(pre_5, out_5))


    L = (l1+ l2 + l3 + l4 + l5) / 5

    return L


#LeTu
def cal_distance_frame(predicted, target):
    assert predicted.shape == target.shape             #shape=(batchsize, num_frames, num_joints, num_coord)=(256,3,21,3)
    number_of_frames = predicted.size(1)     #1,3,5,7
    sum_error = torch.zeros(1).cuda()
    error_all_frames = torch.zeros(1).cuda()

    for i in range(0,number_of_frames-1):            # Frame 
        for j in range(1,16,5):
            #5 fingers
            dist1_frame_pre = torch.norm(predicted[:,i,j] - predicted[:,i,j+5], dim=1)        #length of finger  (1-6), (6-11), (11-16), output.shape=tensor(0.xxxx)
            dist1_frame_after = torch.norm(predicted[:,i+1,j] - predicted[:,i+1,j+5], dim=1)    
            
            dist2_frame_pre = torch.norm(predicted[:,i,j+1] - predicted[:,i,j+6], dim=1)       #length of finger  (2-7), (7-12), (12-17)
            dist2_frame_after = torch.norm(predicted[:,i+1,j+1] - predicted[:,i+1,j+6], dim=1)
            
            dist3_frame_pre = torch.norm(predicted[:,i,j+2] - predicted[:,i,j+7], dim=1)       #length of finger  (3-8), (8-13), (13-18)
            dist3_frame_after = torch.norm(predicted[:,i+1,j+2] - predicted[:,i+1,j+7], dim=1)
            
            dist4_frame_pre = torch.norm(predicted[:,i,j+3] - predicted[:,i,j+8], dim=1)       #length of finger    (4-9), (9-14), (14-19)
            dist4_frame_after = torch.norm(predicted[:,i+1,j+3] - predicted[:,i+1,j+8], dim=1)
            
            dist5_frame_pre = torch.norm(predicted[:,i,j+4] - predicted[:,i,j+9], dim=1)       #length of finger    (5-10), (10-15), (15-20)
            dist5_frame_after = torch.norm(predicted[:,i+1,j+4] - predicted[:,i+1,j+9], dim=1)

            dist1_diff = torch.mean(torch.abs(dist1_frame_after - dist1_frame_pre))   #tensor.Size([batch_size])
            dist2_diff = torch.mean(torch.abs(dist2_frame_after - dist2_frame_pre))
            dist3_diff = torch.mean(torch.abs(dist3_frame_after - dist3_frame_pre))
            dist4_diff = torch.mean(torch.abs(dist4_frame_after - dist4_frame_pre))
            dist5_diff = torch.mean(torch.abs(dist5_frame_after - dist5_frame_pre))
            
            sum_error += dist1_diff + dist2_diff + dist3_diff + dist4_diff + dist5_diff   #tensor.Size([batch_size])
        #Palm
        dist1_palm_pre = torch.norm(predicted[:,i,0] - predicted[:,i,1], dim=1)       #length of connect (0-1)
        dist1_palm_after = torch.norm(predicted[:,i+1,0] - predicted[:,i+1,1], dim=1)
        
        dist2_palm_pre = torch.norm(predicted[:,i,0] - predicted[:,i,2], dim=1)        #length of connect (0-2)
        dist2_palm_after = torch.norm(predicted[:,i+1,0] - predicted[:,i+1,2], dim=1)
        
        dist3_palm_pre = torch.norm(predicted[:,i,0] - predicted[:,i,3], dim=1)         #length of connect (0-3)
        dist3_palm_after = torch.norm(predicted[:,i+1,0] - predicted[:,i+1,3], dim=1)
        
        dist4_palm_pre = torch.norm(predicted[:,i,0] - predicted[:,i,4], dim=1)          #length of connect (0-4)
        dist4_palm_after = torch.norm(predicted[:,i+1,0] - predicted[:,i+1,4], dim=1)
        
        dist5_palm_pre = torch.norm(predicted[:,i,0] - predicted[:,i,5], dim=1)            #length of connect (0-5)
        dist5_palm_after = torch.norm(predicted[:,i+1,0] - predicted[:,i+1,5], dim=1)

        dist1_palm_diff = torch.mean(torch.abs(dist1_palm_after - dist1_palm_pre))
        dist2_palm_diff = torch.mean(torch.abs(dist2_palm_after - dist2_palm_pre))
        dist3_palm_diff = torch.mean(torch.abs(dist3_palm_after - dist3_palm_pre))
        dist4_palm_diff = torch.mean(torch.abs(dist4_palm_after - dist4_palm_pre))
        dist5_palm_diff = torch.mean(torch.abs(dist5_palm_after - dist5_palm_pre))

        sum_error += dist1_palm_diff + dist2_palm_diff + dist3_palm_diff + dist4_palm_diff + dist5_palm_diff
    error_all_frames += sum_error / 20
    return error_all_frames             # diff(frame1-frame0) + diff(frame2-frame1)


relu = nn.ReLU()
import math
def calculate_angle_multiframe(a, b):
    dot_ab = torch.sum(a*b, dim=2, keepdim=True).squeeze()
    length_a = torch.norm(a, dim=2)
    length_b = torch.norm(b, dim=2)
    dot_length = length_a * length_b

    acos = dot_ab/dot_length
    epsilon1 = relu(acos - 0.9999999)
    acos = acos - epsilon1

    epsilon2 = relu(-0.9999999 - acos)
    acos = acos + epsilon2

    angle = torch.arccos(acos)       #radian
    # angle = angle*180/math.pi
    return angle

def calculate_angle_singleframe(a, b):
    dot_ab = torch.sum(a*b, dim=2)
    length_a = torch.norm(a, dim=2)
    length_b = torch.norm(b, dim=2)
    dot_length = length_a * length_b
    acos = dot_ab / dot_length
    epsilon1 = relu(acos - 0.9999999)
    acos = acos - epsilon1

    epsilon2 = relu(-0.9999999 - acos)
    acos = acos + epsilon2
    angle = torch.arccos(acos)       #radian
    # angle = angle*180/math.pi
    return angle





def cal_loss_angle(predicted, target): 

    assert predicted.shape == target.shape      #(batchsize, numframes, numjoints, channels)
    batch_size = predicted.size(0)
    num_frames = predicted.size(1)
    min_theta1 = torch.zeros((batch_size, num_frames)).cuda() + 33*math.pi/180      #if train: shape=(256,3), if test: shape=(256,1)
    max_theta1 = torch.zeros((batch_size, num_frames)).cuda() + 73*math.pi/180 

    min_theta2 = torch.zeros((batch_size, num_frames)).cuda() + 36*math.pi/180 
    max_theta2 = torch.zeros((batch_size, num_frames)).cuda() + 86*math.pi/180 

    min_theta3 = torch.zeros((batch_size, num_frames)).cuda() + 20*math.pi/180 
    max_theta3 = torch.zeros((batch_size, num_frames)).cuda() + 61*math.pi/180 
    relu = nn.ReLU()
    loss_sum = torch.zeros(1).cuda()
    for j in range(1,6,1):            #cal each finger

        wrist = predicted[:,:,0]        #wrist: shape=(batch_size, num_frames, 3) = (256, 3, 3) if train, else (256,2,3)
        mcp = predicted[:,:,j]
        pip = predicted[:,:,j+5]
        dip = predicted[:,:,j+10]
        tip = predicted[:,:,j+15]
        #a,b
        connect_wrist_mcp = wrist - mcp         #shape = (batch_size, num_frames, 3)
        connect_mcp_pip = mcp - pip
        connect_pip_dip = pip - dip
        connect_dip_tip = dip - tip


        
        if num_frames != 1:
            theta1 = calculate_angle_multiframe(connect_wrist_mcp, connect_mcp_pip)     #shape=(batch_size, num_frames)=(256,3)
            
            theta2 = calculate_angle_multiframe(connect_mcp_pip, connect_pip_dip)
            theta3 = calculate_angle_multiframe(connect_pip_dip, connect_dip_tip)
        else:
            theta1 = calculate_angle_singleframe(connect_wrist_mcp, connect_mcp_pip)    #shape=(256,1)
            theta2 = calculate_angle_singleframe(connect_mcp_pip, connect_pip_dip)
            theta3 = calculate_angle_singleframe(connect_pip_dip, connect_dip_tip)
            
        loss_theta1 = relu(min_theta1 - theta1) + relu(theta1 - max_theta1)         #shape=(batch_size, num_frames)
        loss_theta2 = relu(min_theta2 - theta2) + relu(theta2 - max_theta2)
        loss_theta3 = relu(min_theta3 - theta3) + relu(theta3 - max_theta3)
        
        loss_theta1_avg = torch.sum(loss_theta1)
        loss_theta2_avg = torch.sum(loss_theta2)
        loss_theta3_avg = torch.sum(loss_theta3)
        loss_each_finger = loss_theta1_avg + loss_theta2_avg + loss_theta3_avg
    loss_sum = loss_sum + loss_each_finger
    return loss_sum 
#Loss angle between predicted and gt
def cal_loss_direction(predicted, target):
    assert predicted.shape == target.shape
    batch_size = predicted.size(0)
    num_frames = predicted.size(1)
    loss_sum = torch.zeros(1).cuda()

    for j in range(1,6,1):
        wrist_pred = predicted[:,:,0]      #(batch_size, num_frames, coord) = (256, 3, 3)
        wrist_gt = target[:,:,0]

        mcp_pred = predicted[:,:,j]
        mcp_gt = target[:,:,j]

        pip_pred = predicted[:,:,j+5]
        pip_gt = target[:,:,j+5]

        dip_pred = predicted[:,:,j+10]
        dip_gt = target[:,:,j+10]

        tip_pred = predicted[:,:,j+15]
        tip_gt = target[:,:,j+15]

        #a,b predicted
        connect_wrist_mcp_pred = wrist_pred - mcp_pred      #(batch_size, num_frames, coord) = (256, 3, 3)
        connect_mcp_pip_pred = mcp_pred - pip_pred
        connect_pip_dip_pred = pip_pred - dip_pred
        connect_dip_tip_pred = dip_pred - tip_pred

        #a,b gt
        connect_wrist_mcp_gt = wrist_gt - mcp_gt
        connect_mcp_pip_gt = mcp_gt - pip_gt
        connect_pip_dip_gt = pip_gt - dip_gt
        connect_dip_tip_gt = dip_gt - tip_gt

        if num_frames != 1:
            beta1 = calculate_angle_multiframe(connect_wrist_mcp_pred, connect_wrist_mcp_gt)    #(batch_size, num_frames) = (256,3)
            beta2 = calculate_angle_multiframe(connect_mcp_pip_pred, connect_mcp_pip_gt)
            beta3 = calculate_angle_multiframe(connect_pip_dip_pred, connect_pip_dip_gt)
            beta4 = calculate_angle_multiframe(connect_dip_tip_pred, connect_dip_tip_gt)
        else:
            beta1 = calculate_angle_singleframe(connect_wrist_mcp_pred, connect_wrist_mcp_gt)   #(256, 1)
            beta2 = calculate_angle_singleframe(connect_mcp_pip_pred, connect_mcp_pip_gt)
            beta3 = calculate_angle_singleframe(connect_pip_dip_pred, connect_pip_dip_gt)
            beta4 = calculate_angle_singleframe(connect_dip_tip_pred, connect_dip_tip_gt)
            
        loss_sum = beta1 + beta2 + beta3 + beta4
    return torch.mean(loss_sum)

def cal_loss_angle_pred_gt(predicted, target):
    assert predicted.shape == target.shape

    batch_size = predicted.size(0)
    num_frames = predicted.size(1)
    loss_sum = torch.zeros(1).cuda()
    for j in range(1,6,1):
        wrist_pred = predicted[:,:,0]
        wrist_gt = target[:,:,0]

        mcp_pred = predicted[:,:,j]
        mcp_gt = target[:,:,j]

        pip_pred = predicted[:,:,j+5]
        pip_gt = target[:,:,j+5]

        dip_pred = predicted[:,:,j+10]
        dip_gt = target[:,:,j+10]

        tip_pred = predicted[:,:,j+15]
        tip_gt = target[:,:,j+15]

        #a,b predicted
        connect_wrist_mcp_pred = wrist_pred - mcp_pred
        connect_mcp_pip_pred = mcp_pred - pip_pred
        connect_pip_dip_pred = pip_pred - dip_pred
        connect_dip_tip_pred = dip_pred - tip_pred

        #a,b gt
        connect_wrist_mcp_gt = wrist_gt - mcp_gt
        connect_mcp_pip_gt = mcp_gt - pip_gt
        connect_pip_dip_gt = pip_gt - dip_gt
        connect_dip_tip_gt = dip_gt - tip_gt

        if num_frames != 1:
            #Cal theta angle predicted
            theta1_pred = calculate_angle_multiframe(connect_wrist_mcp_pred, connect_mcp_pip_pred)
            theta2_pred = calculate_angle_multiframe(connect_mcp_pip_pred, connect_pip_dip_pred)
            theta3_pred = calculate_angle_multiframe(connect_pip_dip_pred, connect_dip_tip_pred)
            
            # print("connect_mcp_pip_gt: ", connect_mcp_pip_gt[0:4])
            # print("connect_pip_dip_gt: ", connect_pip_dip_gt[0:4])
            #Cal theta angle of gt
            theta1_gt = calculate_angle_multiframe(connect_wrist_mcp_gt, connect_mcp_pip_gt)
            theta2_gt = calculate_angle_multiframe(connect_mcp_pip_gt, connect_pip_dip_gt)
            theta3_gt = calculate_angle_multiframe(connect_pip_dip_gt, connect_dip_tip_gt)
            # print("theta2_gt[0:4]: ",theta2_gt[0:4])

        else:
            theta1_pred = calculate_angle_singleframe(connect_wrist_mcp_pred, connect_mcp_pip_pred)
            theta2_pred = calculate_angle_singleframe(connect_mcp_pip_pred, connect_pip_dip_pred)
            theta3_pred = calculate_angle_singleframe(connect_pip_dip_pred, connect_dip_tip_pred)

            theta1_gt = calculate_angle_singleframe(connect_wrist_mcp_gt, connect_mcp_pip_gt)
            theta2_gt = calculate_angle_singleframe(connect_mcp_pip_gt, connect_pip_dip_gt)
            theta3_gt = calculate_angle_singleframe(connect_pip_dip_gt, connect_dip_tip_gt)
        
        loss_theta1 = torch.norm(theta1_pred - theta1_gt)
        loss_theta2 = torch.norm(theta2_pred - theta2_gt)
        loss_theta3 = torch.norm(theta3_pred - theta3_gt)
        

        loss_sum = loss_theta1 + loss_theta2 + loss_theta3

    return loss_sum
def test_calculation(predicted, target, action, error_sum, data_type, show_protocol2=False):
    """
    get test error sum accumulation
    :param predicted:
    :param target:
    :param action:
    :param action_error_sum:
    :param eva_by_subj:
    :return:
    """
    if data_type == 'FPHAB':
        error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
        if show_protocol2:
            error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    elif data_type.startswith('STB'):
        error_sum = mjmpe_directly(predicted, target, error_sum)

    return error_sum

# def mjmpe_directly(predicted, target,action_error_sum):
#     assert predicted.shape == target.shape
#     num = predicted.size(0)
#     dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
#     # action_error_sum[0] += num
#     # action_error_sum[1] += num * torch.mean(dist).item()
#     action_error_sum.update(torch.mean(dist).item() * num, num)
#     return action_error_sum


def mjmpe_by_action_subject(predicted, target,action,action_error_sum,subject):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
    if len(set(list(action))) == 1 and len(set(list(subject))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name][subject][0] += num
        action_error_sum[action_name][subject][1] += num*torch.mean(dist).item()
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name][subject][0] += 1
            action_error_sum[action_name][subject][1] += dist[i].item()
    return action_error_sum

def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1),dim=len(target.shape) - 2)
    # some action name have space, example: walking 1, change it to walking
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(dist[i].item(), 1)
    return action_error_sum

def mpjpe_by_action_p2(predicted, target,action,action_error_sum):
    """
    Aligned Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred,gt)
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        # action_error_sum[action_name][2] += num*np.mean(dist)
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            # action_error_sum[action_name][2] += dist[i].item()
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
    return action_error_sum


# #
# def p_mpjpe(predicted, target):
#     """
#     Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
#     often referred to as "Protocol #2" in many papers.
#     """
#     assert predicted.shape == target.shape

#     muX = np.mean(target, axis=1, keepdims=True)
#     muY = np.mean(predicted, axis=1, keepdims=True)

#     X0 = target - muX
#     Y0 = predicted - muY

#     normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
#     normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

#     X0 /= normX
#     Y0 /= normY

#     H = np.matmul(X0.transpose(0, 2, 1), Y0)
#     U, s, Vt = np.linalg.svd(H)
#     V = Vt.transpose(0, 2, 1)
#     R = np.matmul(V, U.transpose(0, 2, 1))

#     # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
#     sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
#     V[:, :, -1] *= sign_detR
#     s[:, -1] *= sign_detR.flatten()
#     R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

#     tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

#     a = tr * normX / normY  # Scale
#     t = muX - a * np.matmul(muY, R)  # Translation

#     # Perform rigid transformation on the input
#     predicted_aligned = a * np.matmul(predicted, R) + t

#     # Return MPJPE
#     return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1),axis=len(target.shape) - 2)




# def sym_penalty(dataset,keypoints,pred_out):
#     """
#     get penalty for the symmetry of human body
#     :return:
#     """
#     loss_sym = 0
#     if dataset == 'h36m':
#         if keypoints.startswith('sh'):
#             left_bone = [(0,4),(4,5),(5,6),(8,10),(10,11),(11,12)]
#             right_bone = [(0,1),(1,2),(2,3),(8,13),(13,14),(14,15)]
#         else:
#             left_bone = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13)]
#             right_bone = [(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)]
#         for (i_left,j_left),(i_right,j_right) in zip(left_bone,right_bone):
#             left_part = pred_out[:,:,i_left]-pred_out[:,:,j_left]
#             right_part = pred_out[:, :, i_right] - pred_out[:, :, j_right]
#             loss_sym += torch.mean(torch.norm(left_part, dim=- 1) - torch.norm(right_part, dim=- 1))
#     elif dataset.startswith('STB'):
#         loss_sym = 0
#     return loss_sym

# def bone_length_penalty(dataset,keypoints,pred_out):
#     """
#     get penalty for the consistency of the bone length in sequences
#     :return:
#     """
#     loss_bone = 0
#     if pred_out.size(1) == 1:
#         return 0
#     if dataset == 'h36m' and keypoints is not 'sh_ft_h36m':
#         bone_id = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13),(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)
#             ,(0,7),(7,8),(8,9),(9,10)]
#         for (i,j) in bone_id:
#             bone = torch.norm(pred_out[:,:,i]-pred_out[:,:,j],dim=-1)#N*T
#             loss_bone += torch.sum(torch.var(bone,dim=1))#N

#     return loss_bone






