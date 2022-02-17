import argparse
from ctypes import sizeof
import json
import numpy as np
import os
import pickle
from numpy.core.fromnumeric import shape
from .utils import arse_config



def read_pic(root):
  # 重命名原图
  i=0
  filenames=os.listdir(root)
  print(filenames)
  filenames.sort(key=lambda x:int(x[:-4]))
  for item in filenames:
    pname = os.path.join(root,item)
    nname = os.path.join(root,'{:0>2d}'.format(i)+'.png')
    os.rename(pname,nname)
    i+=1

def json_write_llff(c2w_metrix,h,w,f):
  
  print(type(c2w_metrix)) # list
  print(c2w_metrix[0])
  print(len(c2w_metrix))
  f = f
  hwf = np.array([h,w,f]).reshape(3,1)
  # print("hwf:",hwf)
  # print((hwf).shape)
  c2w_metrix = np.array(c2w_metrix)
  print((c2w_metrix).shape)
  poses = c2w_metrix[:, :3, :4].transpose([1,2,0])
  poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
  poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

  ## 不计算深度
  # poses = poses.transpose([2,0,1])
  # poses = np.reshape(poses,(-1,15))

  save_arr = []
  for i in range(len(c2w_metrix)):
      close_depth, inf_depth = 0,0
      # print( i, close_depth, inf_depth )
      save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
  save_arr = np.array(save_arr)
  print(save_arr[0])

  np.save(os.path.join(os.getcwd(),'poses_bounds.npy'), save_arr)
  print(".npy has been written")


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
    np_s = np.array(data.get('c2w_metrix'))[i]
    dic['transform_matrix'] = np_s.tolist()
    # print(dic)
    frames.append(dic)
  # print(frames)
    print("c2w_metrix:",np_s)
    
  # ca = np.array(data.get('camera_angle_x')).tolist()[0]
  # print('ca:',data.get('camera_angle_x')[0])
  # s = {'camera_angle_x':float(ca[0][0]),'frames':frames}
  s = {'frames':frames}

  js = json.dumps(s,indent=4)
  with open('./jsontrans.json', 'w') as  f:
      f.write(js)     
  print('.json has been written')

