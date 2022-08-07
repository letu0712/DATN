hopenet_pretrained_full_dataset.pkl err: 51 mm
 - Re-train network on full set
 - Train with 21 joints
 - Type: list
 - Sample {'subject': 6,
     'action': 'write',
     'seq': 4,
     'frame': 134,
     'color': '/home/namhv/data/F-PHAB/F-PHAB/Video_files/Subject_6/write/4/color/color_0134.jpeg',
     'hand3d_gt': (21x3), # 3d ground truth
     'hand2d_gt': (21x2), # 2d ground truth
     'hand2d_pred_init': (21x2), #2d predicted ResNet 
     'hand2d_pred': (21x2), #2d predicted ResNet -> GraphNet
     'hand3d_pred': (21x3), #3d predicted ResNet -> GraphNet -> GraphUnet 
     }