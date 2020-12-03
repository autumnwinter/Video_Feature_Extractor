import numpy as np 
import argparse
#from sample_video import sample_from_video
from Feature_Extractor import Feature_Extractor
#parser.add_argument('--inference_video_file', type=str, default=None)
#parser.add_argument('--intput_database_adress', type=str, default=None)
#args = parser.parse_args()
#database = args.intput_database_adress
#inference_video = args.inference_video_file
def inference(frames, database):
  FE = Feature_Extractor()
  inf_feats = FE(frames)
  database = np.load(database)
  

