import json
import numpy as np
import os
import pickle

def json_write():
  data = open(os.path.join(os.getcwd(),'log/c2w_metrix.pkl'),'rb')
  npdata = np.array(pickle.load(data))
  with open('./jsontrans.json', 'w') as  f:
      json.dump(npdata.tolist(),f)     
  print('.json has been written')
