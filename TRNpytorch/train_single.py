#train.py
import argparse
import os
import numpy as np
print (os.getcwd())
import sys
sys.path.append(".")
from sample_video import sample_from_video
#from Feature_Extractor.TRNpytorch.Feature_Extractor import Feature_Extractor
from Feature_Extractor import Feature_Extractor
parser = argparse.ArgumentParser(description="Extract and Index Video Features")
parser.add_argument('--train_videos_file', type=str, default=None)
#parser.add_argument('--class_lables_file', type=str, default=None)
parser.add_argument('--output_database_adress', type=str, default=None)
args = parser.parse_args()
#class_lables_file = args.class_lables_file
train_videos_file = args.train_videos_file
output = args.output_database_adress
#with open(class_lables_file, 'r') as f:
#  lines = f.readlines()
#  number_of_classes = int(lines[0])
#  class_lables = [None] * number_of_classes
#  lines = lines[1:]
#  for line in lines:
#    class_lables[int(line.split(',')[1])] = line.split(',')[0]
videos = []
#lables = []
with open(train_videos_file) as f:
  lines = f.read().splitlines()
  for line in lines:
    videos.append(line)
print("strat indexing videos")
print("Total Train Videos: " + str(len(videos)))
FE = Feature_Extractor()
print("Feature Extractor object has been created")
extracted_features = []
for i,video in enumerate(videos):
  print("Extracting Features From " + str(i+1) + "'th training video from " + str(len(videos)))
  frames = sample_from_video(video, 8)
  features = FE(frames)
  extracted_features.append(features)
extracted_features = np.array(extracted_features)
database = {}
database["features"] = extracted_features
np.save(output, database)