import argparse
import json
import numpy as np
import os
import pickle
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
  data = pickle.load(data)
  # npdata = np.array(data)
  files = sorted(os.listdir(args.folder))


  # 读取文件名
  name = []
  for file in files:
    s = str(os.path.join(args.folder,os.path.splitext(file)[0]))
    name.append(s)
  

  #读取
  
  frames = []
  for i in range(len(name)):
    dic = {}
    dic['file_path'] = name[i]
    # rot = np.array(data.get('rvecs'))[i]
    # dic['rotation'] = rot.tolist()
    np_s = np.array(data.get('c2w_metrix'))[i].T
    dic['transform_matrix'] = np_s.tolist()
    # print(dic)
    frames.append(dic)
  print(frames)
    
  ca = np.array(data.get('ret')).tolist()
  s = {'camera_angle_x':ca,'frames':frames}

  js = json.dumps(s,indent=4)
  with open('./jsontrans.json', 'w') as  f:
      f.write(js)     
  print('.json has been written')

