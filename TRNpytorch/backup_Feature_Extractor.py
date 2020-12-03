

import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy

import torchvision
import torch.nn.parallel
import torch.optim
from models import TSN
import transforms
from torch.nn import functional as F


def extract_frames(video_file, num_frames=8):
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        pass

    output = subprocess.Popen(['ffmpeg', '-i', video_file],
                              stderr=subprocess.PIPE).communicate()
    # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    re_duration = re.compile('Duration: (.*?)\.')
    duration = re_duration.search(str(output[1])).groups()[0]

    seconds = functools.reduce(lambda x, y: x * 60 + y,
                               map(int, duration.split(':')))
    rate = num_frames / float(seconds)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-vf', 'fps={}'.format(rate),
                               '-vframes', str(num_frames),
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    frame_paths = sorted([os.path.join('frames', frame)
                          for frame in os.listdir('frames')])

    frames = load_frames(frame_paths)
    subprocess.call(['rm', '-rf', 'frames'])
    #print(type(frames))
    #print(len(frames))
    #print(frames[0])
    #temp = list((frames[0].getdata()))
    temp = np.asarray(frames[0])
    #print(temp)
    #print(temp.shape)
    #print(len(temp))
    #raise(False)
    return frames


def load_frames(frame_paths, num_frames=8):
    frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames

class Feature_Extractor:
  def __init__(self):
    # options
    #parser = argparse.ArgumentParser(description="test TRN on a single video")
    #group = parser.add_mutually_exclusive_group(required=True)
    #group.add_argument('--video_file', type=str, default=None)
    #group.add_argument('--frame_folder', type=str, default=None)

    #parser.add_argument('--modality', type=str, default='RGB',
                        #choices=['RGB', 'Flow', 'RGBDiff'], )
    #almost constant:
    args_modality = 'RGB'
    args_rendered_output = None
    args_input_size = 224
    args_test_segments = 8
    args_img_feature_dim = 256
    args_consensus_type = "TRNmultiscale"

    #change for investigations
    args_dataset = 'moments'
    args_arch = "InceptionV3"
    args_weights = "pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar"
    args_frame_folder = None
    #parser.add_argument('--dataset', type=str, default='moments',
                        #choices=['something', 'jester', 'moments', 'somethingv2'])

    #parser.add_argument('--rendered_output', type=str, default=None)

    #parser.add_argument('--arch', type=str, default="InceptionV3")
    #parser.add_argument('--input_size', type=int, default=224)
    #parser.add_argument('--test_segments', type=int, default=8)
    #parser.add_argument('--img_feature_dim', type=int, default=256)
    #parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
    #parser.add_argument('--weights', type=str)

    #args = parser.parse_args()

    # Get dataset categories.
    categories_file = 'pretrain/{}_categories.txt'.format(args_dataset)
    categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
    num_class = len(categories)

    args_arch = 'InceptionV3' if args_dataset == 'moments' else 'BNInception'
    self.args_arch = args_arch
    self.args_frame_folder = args_frame_folder 

    # Load model.
    self.net = TSN(num_class,
              args_test_segments,
              args_modality,
              base_model=args_arch,
              consensus_type=args_consensus_type,
              img_feature_dim=args_img_feature_dim, print_spec=False)
    print("The model has been loaded")

    checkpoint = torch.load(args_weights)
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    self.transform = torchvision.transforms.Compose([
    transforms.GroupOverSample(self.net.input_size, self.net.scale_size),
    transforms.Stack(roll=(args_arch in ['BNInception', 'InceptionV3'])),
    transforms.ToTorchFormatTensor(div=(args_arch not in ['BNInception', 'InceptionV3'])),
    transforms.GroupNormalize(self.net.input_mean, self.net.input_std),
    ])
    self.net.load_state_dict(base_dict)
    self.net.cuda().eval()
    
    #print("in bood")

  def __call__(self, frames):
    # # Initialize frame transforms.
    #print("enter to call function")
    from PIL import Image
    args_arch = self.args_arch
    args_frame_folder = self.args_frame_folder


    frames = np.split(frames,frames.shape[0], axis = 0)
    for i in range(len(frames)):
      frames[i] = np.squeeze(frames[i], axis=0)
      frames[i] = Image.fromarray(frames[i])
    transform = self.transform
    data = transform(frames)
    
    input = data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda()

    with torch.no_grad():
        logits, res1, res2 = self.net(input)
        #video_name = args_frame_folder if args_frame_folder is not None else args_video_file
        #np.save((video_name + "_features1"), res1.cpu())
        #np.save((video_name + "_features2"), res2.cpu())
        #np.save((video_name + "_logits"), logits.cpu())

        h_x = torch.mean(F.softmax(logits, 1), dim=0).data
        probs, idx = h_x.sort(0, True)
        result = np.mean(res1.cpu().numpy(), 0)
    torch.cuda.empty_cache()
    return result















# # Obtain video frames
# if args_frame_folder is not None:
#     print('Loading frames in {}'.format(args_frame_folder))
#     import glob
#     # Here, make sure after sorting the frame paths have the correct temporal order
#     frame_paths = sorted(glob.glob(os.path.join(args_frame_folder, '*.jpg')))
#     frames = load_frames(frame_paths)
# else:
#     print('Extracting frames using ffmpeg...')
#     frames = extract_frames(args_video_file, args_test_segments)


# # Make video prediction.
# temp = frames


# #print(len(data_temp))
# #print("transform2:")
# #print(data_temp)
# #print(len(data_temp))
# #print("ok")
# #raise(False)

# data = transform(frames)
# #data_res = data.cpu().detach().numpy()
# #print(type(data_res))
# #print(data_res.shape)
# #from google.colab.patches import cv2_imshow
# #test = data_res[0,:,:]
# #np.save("test_image.npy", test)
# #print(type(test), "type")
# #cv2_imshow(test)
# #cv2.waitKey(0)
# #raise(False)
# #print("inja")
# #print(type(data))
# #print(data.shape)

# input = data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda()

# with torch.no_grad():
#     logits, res1, res2 = net(input)
#     #print(logits.shape)
#     #print(res1.shape)
#     #print(res2.shape)
#     video_name = args_frame_folder if args_frame_folder is not None else args_video_file
#     #np.save((video_name + "_features1"), res1.cpu())
#     #np.save((video_name + "_features2"), res2.cpu())
#     #np.save((video_name + "_logits"), logits.cpu())

#     h_x = torch.mean(F.softmax(logits, 1), dim=0).data
#     probs, idx = h_x.sort(0, True)

# # Output the prediction.
# video_name = args.frame_folder if args.frame_folder is not None else args.video_file
# print('RESULT ON ' + video_name)
# for i in range(0, 5):
#     print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))

# # Render output frames with prediction text.
# if args.rendered_output is not None:
#     prediction = categories[idx[0]]
#     rendered_frames = render_frames(frames, prediction)
#     clip = mpy.ImageSequenceClip(rendered_frames, fps=4)
#     clip.write_videofile(args.rendered_output)
