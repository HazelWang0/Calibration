import json
import numpy as np
import pickle

def json_write():
  data = open('./logs/c2w_metrix.pkl','rb')
  data = pickle.load(data)
  print(type(data))
  npdata = np.array(data)
  print(type(npdata))
  print(npdata.shape)
  data = npdata.tolist()
  with open('./jsontrans.json', 'w') as  f:
      json.dump(data,f)     
