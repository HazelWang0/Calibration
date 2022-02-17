import numpy as np

# 读取.npy文件
arr = np.load("/root/vmip/myProject/nerf/camera/nerfCalibration/poses_bounds.npy")
print(arr[24])
print("load .npy done")