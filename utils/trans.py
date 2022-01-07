import json
import numpy as np
import pickle

def json_write():
  data = open('/root/vmip/myProject/nerf/camera/myCalibration/logs/c2w_metrix.pkl','rb')
  data = pickle.load(data)
  print(type(data))
  npdata = np.array(data)
  print(type(npdata))
  print(npdata.shape)
  data = npdata.tolist()
  with open('./jsontrans.json', 'w') as  f:
      json.dump(data,f)     

  # with open(model_path + "/args.json", 'w') as out:
  #   json.dump(vars(args), out, indent=2, sort_keys=True)