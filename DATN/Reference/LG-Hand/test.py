
import torch.nn as nn
import torch

#Loss angle
relu = nn.ReLU()

def calculate_angle_multiframe(a, b):
    dot_ab = torch.sum(a*b, dim=2, keepdim=True).squeeze()
    length_a = torch.norm(a, dim=2)
    length_b = torch.norm(b, dim=2)

    dot_length = length_a * length_b
    angle = torch.arccos(dot_ab / dot_length)
    angle = angle*180/torch.pi
    return angle

torch.manual_seed(2)

predicted = torch.randn((2,1,21,3))

wrist = predicted[:,:,1]        #shape=(256,3,3)
mcp = predicted[:,:,6]
pip = predicted[:,:,11]
# print("Wrist: ", wrist)
# print("MCP: ", mcp)
# print("PIP: ", pip)
connect_wrist_mcp = wrist - mcp         #(256,3)
connect_mcp_pip = mcp - pip             #
# print("Connect_wrist_mcp: ", connect_wrist_mcp)
# print("Connect_mcp_pip: ", connect_mcp_pip)

dot_ab = connect_wrist_mcp * connect_mcp_pip
# print("dot_ab: ", dot_ab)
dot_ab = torch.sum(dot_ab, dim=2, keepdim=True).squeeze()                 # a*b   shape=(2,3,1)
# print("sum__each_row:", dot_ab)                                 

length_connect_wrist_mcp = torch.norm(connect_wrist_mcp, dim=2) 
length_connect_mcp_pip = torch.norm(connect_mcp_pip, dim=2)

# print("length_connect_wrist_mcp: ", length_connect_wrist_mcp)
# print("length_connect_mcp_pip: ", length_connect_mcp_pip)
dot_length = length_connect_mcp_pip * length_connect_wrist_mcp     # |a|*|b|
# print(dot_length.squeeze())

angle = torch.arccos(dot_ab / dot_length.squeeze())   #shape = (batchsize, frame)   #radian
angle = angle*180/torch.pi
# print("Angle: ", angle)

# print(angle)
# batch_size = predicted.size(0)
# num_frames = predicted.size(1)
# min_theta1 = torch.zeros((batch_size, num_frames)) + 30
# max_theta1 = torch.zeros((batch_size, num_frames)) + 70


# relu = nn.ReLU()

# loss = relu(min_theta1 - angle) + relu(angle - max_theta1)
# loss_avg_batch = torch.sum(loss) / batch_size
# loss_avg_frame = loss_avg_batch / num_frames
# print(loss_avg_frame)

def calculate_angle_singleframe(a, b):
    dot_ab = torch.sum(a*b, dim=2)
    length_a = torch.norm(a, dim=2)
    length_b = torch.norm(b, dim=2)
    dot_length = length_a * length_b
    angle = torch.arccos(dot_ab / dot_length)
    angle = angle*180/torch.pi
    return angle

def cal_loss_angle(predicted, target): 
    assert predicted.shape == target.shape      #(batchsize, numframes, numjoints, channels)

    batch_size = predicted.size(0)
    num_frames = predicted.size(1)
    min_theta1 = torch.zeros((batch_size, num_frames)).cuda() + 33
    max_theta1 = torch.zeros((batch_size, num_frames)).cuda() + 73

    min_theta2 = torch.zeros((batch_size, num_frames)).cuda() + 36
    max_theta2 = torch.zeros((batch_size, num_frames)).cuda() + 86

    min_theta3 = torch.zeros((batch_size, num_frames)).cuda() + 20
    max_theta3 = torch.zeros((batch_size, num_frames)).cuda() + 61

    relu = nn.ReLU()

    loss_sum = torch.zeros(1).cuda()
    for j in range(1,6,1):
        wrist = predicted[:,:,0]
        mcp = predicted[:,:,j]
        pip = predicted[:,:,j+5]
        dip = predicted[:,:,j+10]
        tip = predicted[:,:,j+15]
        print("wrist: ", wrist)

        #a,b
        connect_wrist_mcp = wrist - mcp
        connect_mcp_pip = mcp - pip
        connect_pip_dip = pip - dip
        connect_dip_tip = dip - tip
        print("connect_wrist_mcp: ", connect_wrist_mcp)
        print("connect_mcp_pip: ", connect_mcp_pip)
        
        if num_frames != 1:
            theta1 = calculate_angle_multiframe(connect_wrist_mcp, connect_mcp_pip)
            theta2 = calculate_angle_multiframe(connect_mcp_pip, connect_pip_dip)
            theta3 = calculate_angle_multiframe(connect_pip_dip, connect_dip_tip)
        else:
            theta1 = calculate_angle_singleframe(connect_wrist_mcp, connect_mcp_pip)
            theta2 = calculate_angle_singleframe(connect_mcp_pip, connect_pip_dip)
            theta3 = calculate_angle_singleframe(connect_pip_dip, connect_dip_tip)
        print("theta1.shape: ", theta1.shape)

        loss_theta1 = relu(min_theta1 - theta1) + relu(theta1 - max_theta1)
        loss_theta2 = relu(min_theta2 - theta2) + relu(theta2 - max_theta2)
        loss_theta3 = relu(min_theta3 - theta3) + relu(theta3 - max_theta3)
        print("loss_theta1: ", loss_theta1)
        loss_theta1_avg = torch.sum(loss_theta1)
        loss_theta2_avg = torch.sum(loss_theta2)
        loss_theta3_avg = torch.sum(loss_theta3)
        print("loss_theta1_avg: ", loss_theta1_avg)

        loss_each_finger = loss_theta1_avg + loss_theta2_avg + loss_theta3_avg
    
    loss_sum = loss_sum + loss_each_finger
    return loss_sum / 15

print(cal_loss_angle(predicted=predicted, target=predicted))

# print(predicted)
wrist = predicted[:,:,0]
mcp = predicted[:,:,1]
pip = predicted[:,:,6]
print("wrist.shape: ", wrist.shape)
print(wrist)

connect_wrist_mcp = wrist - mcp
connect_mcp_pip = mcp - pip

print("connect_wrist_mcp: ",connect_wrist_mcp)
print("connect_mcp_pip: ",connect_mcp_pip)

print(connect_wrist_mcp*connect_mcp_pip)
dot_ab = torch.sum(connect_wrist_mcp*connect_mcp_pip, dim=2)
print("dot_ab: ", dot_ab.shape)
length_a = torch.norm(connect_wrist_mcp, dim=2)
length_b = torch.norm(connect_mcp_pip, dim=2)
print("length_a: ", length_a)
dot_length = length_a * length_b

angle = torch.arccos(dot_ab/ dot_length)
angle = angle*180/torch.pi
print("angle: ",angle)