from tqdm import tqdm
from utils.utils1 import *
import data.common.eval_cal as eval_cal 
import torch
import numpy as np 
import random


#Theshold=100
#71.28 : LG-Hand
#63.14 : SST-GCN
#60.09 : ST-GCN 

def step(split, opt, actions, dataLoader, model, criterion, optimizer=None):#, key_index=None

    # initialize some definitions
    num_data_all = 0
    num_correct = [0 for i in range(0,250)]
    thresholds = [i for i in range(0,250)]

    
    abc = []

    loss_all_sum = {'loss_gt':AccumLoss(), 
                    'loss_diff':AccumLoss(),
                    'loss_dist':AccumLoss(), 
                    'loss_sum':AccumLoss(),
                    'loss_thumb':AccumLoss(), 
                    'loss_index':AccumLoss(), 
                    'loss_middle':AccumLoss(), 
                    'loss_ring':AccumLoss(), 
                    'loss_pinky':AccumLoss(), 
                    'loss_Wrist':AccumLoss(), 
                    'loss_MCP':AccumLoss(), 
                    'loss_PIP':AccumLoss(), 
                    'loss_DIP':AccumLoss(), 
                    'loss_TIP':AccumLoss(), 
                    'loss_angle_pred_gt':AccumLoss(),
                    'loss_direction':AccumLoss()}

    mean_error = {'xyz': 0.0, 'post': 0.0}
    error_sum = AccumLoss()


    if opt.dataset == 'FPHAB':
        limb_Wrist = [0] 
        limb_PIP = [6,7,8,9,10] 
        limb_TIP = [16,17,18,19,20]
        action_error_sum = define_error_list(actions)

        action_error_sum_post_out = define_error_list(actions)
 


    criterion_mse = criterion['MSE']
    model_st_gcn = model['st_gcn']
    model_post_refine = model['post_refine']


    if split == 'train':
        model_st_gcn.train()
        if opt.out_all:                 #output 1 frame or all frames
            out_all_frame = True       #output all frames
        else:
            out_all_frame = False      #output 1 frames

    else:
        model_st_gcn.eval()
        out_all_frame = False           #output 1 frames

    torch.cuda.synchronize()

    #loss during training
    loss_sum_epoch = 0
    loss_direction_epoch = 0
    loss_gt_epoch = 0 
    loss_diff_epoch = 0
    loss_post_refine_epoch = 0
    loss_angle_pred_gt_epoch = 0
    # load data
    for i, data in enumerate(tqdm(dataLoader, 0)):
        

        if split == 'train':
            if optimizer is None:
                print("error! No Optimizer")
            else:
                optimizer_all = optimizer

        # load and process data
        batch_cam, gt_3D, input_2D, action, subject, scale , bb_box, cam_ind = data
        """
            gt_3D.shape = (batch_size, number of frame, number of joints, number of coordinates in 3D) = (256,3,21,3)
            input_2D = (batch_size, number of frame, number of joints, number of coordinates in 2D) = (256,3,21,2)
            len(action) = (batch_size) = 256
        """
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe (split,[input_2D, gt_3D, batch_cam, scale, bb_box])
        N = input_2D.size(0)    #batchsize=256


        num_data_all += N    # number of data

        out_target = gt_3D.clone().view(N, -1, opt.n_joints, opt.out_channels)    #(256,3,21,3) with length sequence =3
        # out_target[:, :, 0] = 0
        gt_3D = gt_3D.view(N, -1, opt.n_joints, opt.out_channels).type(torch.cuda.FloatTensor)    #(256,3,21,3) to cuda

        if out_target.size(1) > 1:        #if number of frames > 1   (3,5,7)
            out_target_single = out_target[:,opt.pad].unsqueeze(1)    #Target frame: If sequence frame=3 => target frame is frame index 1 (0,1,2)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:                                   #with case 1 frame
            out_target_single = out_target        
            gt_3D_single = gt_3D
        

        # start forward process
        #input_2D.shape = (256,3,21,2) = (N,T,V,C)
    
        input_2D = input_2D.view(N, -1, opt.n_joints,opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor) # N, C, T, V, M = (256, 2, 3, 21, 1): Input Model (line 15 st_gcn_multi_frame.py)

        output_3D = model_st_gcn(input_2D, out_all_frame)

        output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.n_joints, opt.out_channels)     #(256,3,21,3)
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),opt.n_joints, opt.out_channels)    #(256,3,21,3)
        if output_3D.size(1) > 1:
            output_3D_single = output_3D[:, opt.pad].unsqueeze(1)      #output_3D_single.shape=(256,1,21,3)
        else:
            output_3D_single = output_3D        #(256,1,21,3)

        #If train: pred_out.shape = (256,3,21,3)     3 frames
        #If test:  pred_out.shape = (256,1,21,3)     1 frames (mid frame (0,1,2) => frame 1)
        if split == 'train':
            pred_out = output_3D            
        elif split == 'test':
            pred_out = output_3D_single


            #Number correct data (> threshold)
            # for i in range(output_3D_single.size(0)):
            #     for threshold in thresholds:
            #         if torch.norm(pred_out[i,:,:,:] - gt_3D_single[i,:,:,:]).item() < threshold:
            #             num_correct[threshold] += 1
        
        # from uvd get xyz
        input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints ,2)    #(256,3,21,2)


        if opt.crop_uv:
            pred_uv = back_to_ori_uv(input_2D, bb_box)
        else:
            pred_uv = input_2D

        uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
        xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
        # xyz[:, :, 0, :] = 0

        if opt.post_refine:
            post_out = model_post_refine(output_3D_single, xyz)
            loss_post_refine = eval_cal.mpjpe(post_out, out_target_single)
        else:
            loss_post_refine = torch.zeros(1).cuda()


        #calculate loss
        """
            pred_out.shape = (batchsize, number of frames, number of joints, number of coordinates)=(256,3,21,3)
            out_target.shape = pred_out.shape = (256,3,21,3)
        """
        loss_gt = eval_cal.mpjpe(pred_out, out_target)

        loss_dist = eval_cal.cal_distant(pred_out,out_target)
        #loss_dist = torch.zeros(1).cuda()

        # #LeTu

        
        loss_angle_pred_gt = eval_cal.cal_loss_angle_pred_gt(pred_out, out_target)
        #loss_angle_pred_gt = torch.zeros(1).cuda()
        
        loss_direction = eval_cal.cal_loss_direction(pred_out, out_target)
        #loss_direction = torch.zeros(1).cuda()
        

            
        #loss diff
        #if only 1 frame => loss_diff = 0    
        if opt.pad == 0 or split == 'test':
            loss_diff = torch.zeros(1).cuda()
        else:
            weight_diff = 0.3 * torch.ones(output_3D[:, :-1, :].size()).cuda()
            weight_diff[:, :, limb_Wrist] = 0.5
            weight_diff[:, :, limb_PIP] = 0.3
            weight_diff[:, :, limb_TIP] = 0.1
            
            diff = (output_3D[:,1:] - output_3D[:,:-1]) * weight_diff
            loss_diff = criterion_mse(diff, Variable(torch.zeros(diff.size()), requires_grad=False).cuda())



        loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_dist * loss_dist + opt.co_angle_pred_gt * loss_angle_pred_gt + opt.co_direction * loss_direction

        loss_sum_epoch = loss_sum_epoch + loss.detach().cpu().item() * N

        loss_gt_epoch = loss_gt_epoch + loss_gt.detach().cpu().item() * N
        loss_diff_epoch = loss_diff_epoch + loss_diff.detach().cpu().item() * N
        loss_post_refine_epoch = loss_post_refine_epoch + loss_post_refine.detach().cpu().item() * N
        loss_angle_pred_gt_epoch = loss_angle_pred_gt_epoch + loss_angle_pred_gt.detach().cpu().item() * N
        loss_direction_epoch = loss_direction_epoch + loss_direction.detach().cpu().item() * N

        loss_all_sum['loss_gt'].update(loss_gt.detach().cpu().numpy() * N, N)
        loss_all_sum['loss_sum'].update(loss.detach().cpu().numpy() * N, N)
        loss_all_sum['loss_dist'].update(loss_dist.detach().cpu().numpy() * N, N)
        loss_all_sum['loss_diff'].update(loss_diff.detach().cpu().numpy() * N, N)

        #LeTu
        loss_all_sum['loss_angle_pred_gt'].update(loss_angle_pred_gt.detach().cpu().numpy() * N, N)
        loss_all_sum['loss_direction'].update(loss_direction.detach().cpu().numpy() * N, N)
        
        # Loss theo cac ngon tay va loai khop tay
        Thumb_target = out_target[:,:,1::5]    #1::5 = 1,6,11,16  (start at 1 with step = 5)
        Thumb_pre = pred_out[:,:,1::5]

        Index_target = out_target[:,:,2::5]
        Index_pre = pred_out[:,:,2::5]

        Middle_target = out_target[:,:,3::5]
        Middle_pre = pred_out[:,:,3::5]

        Ring_target = out_target[:,:,4::5]
        Ring_pre = pred_out[:,:,4::5]

        Pinky_target = out_target[:,:,5::5]
        Pinky_pre = pred_out[:,:,5::5]
        
        
        # Moi khop cua ban tay
        Wrist_target = out_target[:,:,:1]
        Wrist_pre = pred_out[:,:,:1]

        MCP_target = out_target[:,:,1:6]
        MCP_pre = pred_out[:,:,1:6]

        PIP_target = out_target[:,:,6:11]
        PIP_pre = pred_out[:,:,6:11]

        DIP_target = out_target[:,:,11:16]
        DIP_pre = pred_out[:,:,11:16]

        TIP_target = out_target[:,:,16:21]
        TIP_pre = pred_out[:,:,16:21]

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

        # train backpropogation
        if split == 'train':
            torch.autograd.set_detect_anomaly(True)
            optimizer_all.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model_st_gcn.parameters(), 5)

            optimizer_all.step()
            # pred_out[:,:,0,:] = 0
            joint_error = eval_cal.mpjpe(pred_out,out_target).item()
            error_sum.update(joint_error*N, N)



        
        elif split == 'test':
            # pred_out[:, :, 0, :] = 0
            action_error_sum = eval_cal.test_calculation(pred_out, out_target, action, action_error_sum,
                                                         opt.dataset, show_protocol2=opt.show_protocol2)      
            if opt.post_refine:
                # post_out[:, :, 0, :] = 0
                action_error_sum_post_out = eval_cal.test_calculation(post_out, out_target, action, action_error_sum_post_out, opt.dataset,
                                                             show_protocol2=opt.show_protocol2)

                    
    if split == 'train':
        mean_error['xyz'] = error_sum.avg
        print('loss dist each frame of 1 sample: %f mm' % (loss_all_sum['loss_dist'].avg))
        print('loss gt each frame of 1 sample: %f mm' % (loss_all_sum['loss_gt'].avg))
        print('loss of 1 sample: %f' % (loss_all_sum['loss_sum'].avg))
        print('loss diff of 1 sample: %f' % (loss_all_sum['loss_diff'].avg))
        print('mean joint error: %f' % (mean_error['xyz']))
        
        if opt.post_refine:
            with open("error_training_refine.txt", "a") as file:
                file.write(str(mean_error['xyz'])+"\n")
        else:
            with open("error_training.txt", "a") as file:
                file.write(str(mean_error['xyz'])+"\n")


    elif split == 'test':
        if not opt.post_refine:
            dis = eval_cal.cal_distant(pred_out,out_target)


            print(dis)
            mean_error_all = print_error(opt.dataset, action_error_sum, opt.show_protocol2)

            mean_error['xyz'] = mean_error_all

            print('loss gt each frame of 1 sample: %f mm' % (loss_all_sum['loss_gt'].avg))
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
            
                      

        elif opt.post_refine:
            print('-----post out')
            mean_error_all = print_error(opt.dataset, action_error_sum_post_out,  opt.show_protocol2)
            mean_error['post'] = mean_error_all

            print('loss gt each frame of 1 sample: %f mm' % (loss_all_sum['loss_gt'].avg))
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
                             
    loss_sum_epoch = loss_sum_epoch / num_data_all
    loss_gt_epoch = loss_gt_epoch / num_data_all
    loss_diff_epoch = loss_diff_epoch / num_data_all
    loss_post_refine_epoch = loss_post_refine_epoch / num_data_all
    loss_angle_pred_gt_epoch = loss_angle_pred_gt_epoch / num_data_all
    loss_direction_epoch = loss_direction_epoch / num_data_all
    if opt.post_refine:
        with open("loss_running_refine.txt", "a") as file:
            file.write(str(loss_sum_epoch)+","+str(loss_gt_epoch)+","+str(loss_diff_epoch)+","+str(loss_post_refine_epoch)+","+str(loss_angle_pred_gt_epoch)+","+str(loss_direction_epoch)+"\n")
    else:
        with open("loss_running.txt", "a") as file:
            file.write(str(loss_sum_epoch)+","+str(loss_gt_epoch)+","+str(loss_angle_pred_gt_epoch)+","+str(loss_direction_epoch)+"\n")

    # if split == 'test':
    #     print("Num correct prediction: ", num_correct)
    #     print("Num data: ", num_data_all)


    #     result = [correct/num_data_all for correct in num_correct]
    #     with open("result2.txt", "a") as file:
    #         file.write(str(result))
    return mean_error


def train(opt, actions,train_loader,model, criterion, optimizer):
    return step('train',  opt,actions,train_loader, model, criterion, optimizer) # Train with sequence != 1,3


def val( opt, actions,val_loader, model, criterion):
    return step('test',  opt,actions,val_loader, model, criterion)     # Test with sequence = 1,3