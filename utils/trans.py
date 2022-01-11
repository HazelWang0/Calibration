import argparse
import json
import numpy as np
import os
import pickle
import glob

from numpy.core.fromnumeric import shape
from .utils import arse_config



def read_pic(root):
  # 重命名原图
  i=0
  for item in os.listdir(root):
    pname = os.path.join(root,item)
    nname = os.path.join(root,'{:0>2d}'.format(i)+'.png')
    os.rename(pname,nname)
    i+=1


def json_write():
  args = arse_config()
  data = open(os.path.join(os.getcwd(),'log/c2w_metrix.pkl'),'rb')
  npdata = np.array(pickle.load(data))
  files = sorted(os.listdir(args.folder))

  name = []
  for file in files:
    s = str(os.path.join(args.folder,os.path.splitext(file)[0]))
    name.append(s)

  dic = {}
  frames = []
  for i in range(len(name)):
    dic['file_path'] = name[i]
    np_s = npdata[i].T
    dic['transform_matrix'] = np_s.tolist()
    frames.append(dic)
  s = {'camera_angle_x':11,'frames':frames}

  js = json.dumps(s,indent=4)
  with open('./jsontrans.json', 'w') as  f:
      f.write(js)     
  print('.json has been written')

