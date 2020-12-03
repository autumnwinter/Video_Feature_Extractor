import numpy as np 
from Feature_Extractor import Feature_Extractor
from scipy.spatial import distance
#from scipy.special import softmax

def softmax(x):
    #print(type(x), "in softmax function")
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def inference(FE,frames, database, class_lables):
  
  #FE = Feature_Extractor()
  inf_feats = FE(frames)
  
  database = np.load(database,allow_pickle= True).item()
  train_features = database['features']
  #KNN:  (without using faiss)
  #will be implemented using faiss

  #coppied from train.py file, will be implemented as a function:
  #(reading class lables)
  with open(class_lables, 'r') as f:
    lines = f.readlines()
  number_of_classes = int(lines[0])
  class_lables = [None] * number_of_classes
  lines = lines[1:]
  class_scores = [0] * number_of_classes
  for line in lines:
    class_lables[int(line.split(',')[1])] = line.split(',')[0]
  
  for i,category in enumerate(train_features):
    total_dis = 0
    for video in category:
      dis = distance.euclidean(inf_feats, video)
      #print(dis)
      total_dis += dis
    if len(category) != 0:
      class_scores[i] = total_dis / len(category)
  class_scores = np.array(class_scores)
  class_scores = np.max(class_scores) - class_scores
  #print(class_scores)
  #print(class_scores.shape)
  #print(type(class_scores))
  #print(type(class_scores[0]))
  
  class_probs = softmax(class_scores)
  #class_probs = round(class_probs, 2)
  result = {}
  for i,label in enumerate(class_lables):
    result[label] = class_probs[i]
  return(result)

