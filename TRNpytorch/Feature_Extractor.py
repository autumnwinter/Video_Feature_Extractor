import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy
from PIL import Image
import numpy as np
import torchvision
import torch.nn.parallel
import torch.optim
from models import TSN
import transforms
from torch.nn import functional as F


class Feature_Extractor:
  def __init__(self):


    #setting the arguments of Feature Extractor

    arguments = {}
    arguments['modality'] = 'RGB'
    arguments['rendered_output'] = None
    arguments['input_size'] = 224
    arguments['test_segments'] = 8
    arguments['img_feature_dim'] = 256
    arguments['consensus_type'] = "TRNmultiscale"
    arguments['dataset'] = 'moments'
    arguments['arch'] = "InceptionV3"
    arguments['weights'] = "pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar"
    arguments['frame_folder'] = None
    arguments['arch'] = 'InceptionV3' if arguments['dataset'] == 'moments' else 'BNInception'
    self.args = arguments
    

    #reading number of classes
    categories_file = 'pretrain/{}_categories.txt'.format(arguments['dataset'])
    categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
    num_class = len(categories)


    # Load model.
    self.net = TSN(num_class,
              arguments['test_segments'],
              arguments['modality'],
              base_model=arguments['arch'],
              consensus_type=arguments['consensus_type'],
              img_feature_dim=arguments['img_feature_dim'], print_spec=False)
    

    print("The model has been loaded")




    checkpoint = torch.load(arguments['weights'])
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    
    #creating data transform
    self.transform = torchvision.transforms.Compose([
    transforms.GroupOverSample(self.net.input_size, self.net.scale_size),
    transforms.Stack(roll=(arguments['arch'] in ['BNInception', 'InceptionV3'])),
    transforms.ToTorchFormatTensor(div=(arguments['arch'] not in ['BNInception', 'InceptionV3'])),
    transforms.GroupNormalize(self.net.input_mean, self.net.input_std),
    ])
    self.net.load_state_dict(base_dict)
    self.net.cuda().eval()
    

  def __call__(self, frames):

    arch = self.args['arch']
    frame_folder = self.args['frame_folder']
    frames = np.split(frames,frames.shape[0], axis = 0)
    for i in range(len(frames)):
      frames[i] = np.squeeze(frames[i], axis=0)
      frames[i] = Image.fromarray(frames[i])
    transform = self.transform
    data = transform(frames)
    
    input = data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda()

    with torch.no_grad():
        _, features, _ = self.net(input)
        features = np.mean(features.cpu().numpy(), 0)
    torch.cuda.empty_cache()
    return features