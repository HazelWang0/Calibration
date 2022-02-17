import cv2
import numpy as np
import os
import sys
import glob
import pickle
import math
from .utils import arse_config
from .trans import read_pic, read_pic


objpoints=[]
imgpoints=[]


def draw(c2w,objp):
    print('objp',objp)
    M2  = np.trunc(10*c2w)
    print('M2',M2)
    ################相机内参数矩阵################
    f=8
    dx=0.01
    dy=0.01
    u0=640
    v0=480
    list1=[f/dx,0,u0,0,0,f/dy,v0,0,0,0,1,0]
    M1=np.mat(list1).reshape(3,4)
    ########################创建空白图像############################################
    img= np.zeros((240*2,320*2,3), np.uint8) 
    img.fill(255)
    ##########对每个点进行透视运算，将世界坐标转换为像素坐标，并将其标记在空白图像中##############
    corner_out=np.zeros((10*10,2),np.float32)
    print(corner_out[0])
    k=0
    sigma=0.12
    print('M1',M1)
    for l in objp:
        
        l=np.append(l,1)
        l=np.mat(l)

        out=(M1*M2*l.T)/((M2*l.T)[2,0])
        corner_out[k][0]=float(out[0,0])
        corner_out[k][1]=float(out[1,0])
    
        cv2.circle(img,(corner_out[k][0],corner_out[k][1]),1,(0,0,255),4)
        k+=1

    np.save('corner.npy',corner_out)
    cv2.imwrite('output.jpg',img)
    cv2.imshow('image', img)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

    # corner = tuple(corners[0].ravel())
    # img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    # img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    # img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    # return img


# 获得像素坐标
def calibration_photo(rvecs,tvecs,w1,h1):
    tvecs = np.array(tvecs).reshape(3,1)
    rvecs = np.array(rvecs).reshape(3,1)
    objp = np.zeros((w1*h1,3), np.float32)
    objp[:,:2] = np.mgrid[0:w1,0:h1].T.reshape(-1,2)
    objp = objp*18.1  # 18.1 mm

    list1 = rvecs
    in_site=np.mat(list1)
    in_rr=in_site/180*math.pi

    print('tvecs',tvecs)
    print('rvecs',rvecs)

    #旋转向量转化为旋转矩阵
    in_r=cv2.Rodrigues(in_rr,jacobian=0)[0]

    #获得外参数矩阵
    list2 = tvecs
    list2 = np.vstack((in_r,list2.reshape(1,3)))

    #列合并
    yi=np.mat([0,0,0,1])
    c2w_metrix=np.hstack((list2,yi.T))
    # print('c2w_metrix:',c2w_metrix)
   
    return(c2w_metrix,objp)

# # 计算位姿
# def calibration(ret,mtx,dist,rvecs,tvecs,corners,w1,h1):
#     tvecs = np.array(tvecs).reshape(3,1)
#     rvecs = np.array(rvecs).reshape(3,1)
#     objp = np.zeros((w1*h1,3), np.float32)
#     objp[:,:2] = np.mgrid[0:w1,0:h1].T.reshape(-1,2)
#     objp = objp*18.1  # 18.1 mm

#     list1 = rvecs
#     in_site=np.mat(list1)
#     in_rr=in_site/180*math.pi

#     # 找到图像平面点角点坐标
    
#     ret =True

#     if ret:
#         _,R,T=cv2.solvePnP(objp,corners,mtx,dist)
#         print('所求结果：')
#         print("旋转向量",R)
#         print("平移向量",T)



def get_inner_mtx(photo_path,w1,h1):
    # number of coners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
    objp = np.zeros((w1*h1,3), np.float32)
    objp[:,:2] = np.mgrid[0:w1,0:h1].T.reshape(-1,2)
    objp = objp*18.1  # 18.1 mm
    img = cv2.imread(photo_path)
    # print(img.shape)
    #获取画面中心点
    #图像的长宽(480, 640, 3)
    print()
    h, w = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    u, v = img.shape[:2]
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w1,h1), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

    #标定
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return(ret,mtx,dist,rvecs,tvecs,corners)


def set_calibration():
    args = arse_config()
    read_pic(args.folder)
    root = os.path.join(args.folder,'*.png') # 标定图像保存路径
    photos_path = glob.glob(root)
    w1 = args.config[0]
    h1 = args.config[1] 
    c2w_metrix = []
    print('calibrating')
    for photo_path in photos_path:
        # mtx相机内参，dist相机畸变
        ret,mtx,dist,rvecs,tvecs,corners = get_inner_mtx(photo_path,w1,h1)
        c2w,objp = calibration_photo(rvecs[-1],tvecs[-1],w1,h1)
        draw(c2w,objp)
        c2w_metrix.append(c2w)
        





    # os.makedirs(os.path.join(os.getcwd(),'log'),exist_ok=True)
    # f = os.path.join(os.getcwd(),'log','c2w_metrix.pkl')
    # log = {'c2w_metrix':c2w_metrix,'ret':ret,'rvecs':rvecs}
    # with open(f, 'wb') as file:
    #     pickle.dump(log, file)
    # print('finish calibrating')

