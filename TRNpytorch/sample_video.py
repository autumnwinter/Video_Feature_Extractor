import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

def sample_from_video(adress, num_frames):
  cap = cv.VideoCapture(adress)
  length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  length_of_segment = np.floor(length/num_frames)
  sampling_point = np.floor(length_of_segment / 2)
 #print(length_of_segment)
  sampled_frames =[]
  for frame_number in range(length):
    ret, frame = cap.read()
    if frame_number == sampling_point:
      sampled_frames.append(frame)
      sampling_point += length_of_segment
  return np.array(sampled_frames)