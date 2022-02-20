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

def show(c2w,cp):
    # 创建画布
    fig = plt.figure(figsize=(5, 5),
                    facecolor='whitesmoke'
                    )
    # 创建 3D 坐标系
    ax = fig.gca(fc='whitesmoke',
                projection='3d' 
                )
    # -------------------------------- 绘制 3D 图形 --------------------------------
    # # 二元函数定义域平面
    # x = np.linspace(0, 9, 9)
    # y = np.linspace(0, 9, 9)
    # X, Y = np.meshgrid(x, y)
    # # 平面 z=4.5 的部分
    # ax.plot_surface(X,
    #             Y,
    #             Z=X*0+4.5,
    #             color='g',
    #             alpha=0.6
    #            ) 
    # # 散点图
    dots = np.array(c2w).astype(int) # list,在画图前必须转为int类型
    print(dots.shape[0])
    for i in range(dots.shape[0]):
        dot = dots[i]
        print('dot:',dot)
        print('i:',i)
        print('dot[0][0]:',dot[0][0])
        print('dot[1]:',dot[1][0])


        ax.scatter(dot[0],dot[1],dot[2],c='g', marker='o')
        ax.text(dot[0][3],dot[1][3],dot[2][3],i,fontsize=12, color = "r", style = "italic")
    # --------------------------------  --------------------------------
    # 设置坐标轴标题和刻度
    ax.set(xlabel='X',
        ylabel='Y',
        zlabel='Z',
        xlim=(0, 9),
        ylim=(0, 9),
        zlim=(0, 9),
        xticks=np.arange(-200, 400, 60),
        yticks=np.arange(-200, 400, 60),
        zticks=np.arange(-600, 0, 60)
        )

    # 调整视角
    ax.view_init(elev=15,    # 仰角
                azim=60   # 方位角
                )

    # 显示图形
    plt.show()

def showsurface(c2w,cp,rvecs,tvecs):
    fig = plt.figure(figsize=(5, 5),
                    facecolor='whitesmoke'
                    )
    # 创建 3D 坐标系
    ax = fig.gca(fc='whitesmoke',
                projection='3d' 
                )
    c2w = np.array(c2w)
    cp = np.array(cp)
    print('shape c2w:',c2w.shape)
    print('shape cp:',cp.shape)
    print('c2w[0]:',c2w[0])
    print('cp[0]:',cp[0])
    for i in range(cp.shape[0]):
        print('i:',i)
        print('cp[i][0]:',cp[i][0])
        print('cp.shape[0]:',cp.shape[0] )
        x,y,z = np.meshgrid(list(cp[i][0]),list(cp[i][1]),list(cp[i][2]))
        # ax.scatter(x,y,z,c='g', marker='o')

        print('x:',x)
        print('y:',y)
        print('z:',z)
        print('c2w[i]:',i)
        # u = c2w[i][0]
        # v = c2w[i][1]
        # w = c2w[i][2]
        # u, v, w = np.meshgrid(np.dot(cp[i],c2w[i]))
        # u = np.dot(cp[i],c2w[i])[0]
        # print('u:',u)
        print('(np.dot(cp[i],c2w[i]):',(cp[i]*rvecs[i])+tvecs[i])
        w2c = (cp[i]*rvecs[i])+tvecs[i]
        u = w2c*cp[i][0][0]
        v = w2c*cp[i][1][0]
        w = w2c*cp[i][2][0]
        print('w2c*cp[i]：',w2c*cp[i] )
        print('type:',type(w2c*cp[i]))
        print('w2c*cp[i][0][0]：',w2c*cp[i][0][0] )

        # ax.scatter(u, v, w,c='g', marker='o')

        ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
    
    plt.show()
 


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

    #旋转向量转化为旋转矩阵
    in_r=cv2.Rodrigues(in_rr,jacobian=0)[0]

    R=in_r
    t=tvecs
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    w2c_metrix=np.concatenate([np.concatenate([R, t], 1), bottom], 0)
    print('w2c_metrix:',w2c_metrix)
    c2w_metrix = np.linalg.inv(w2c_metrix)
    print('c2w_metrix:',c2w_metrix)
    return(c2w_metrix)


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
    return(ret,mtx,dist,rvecs,tvecs,corners,h,w)


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
        ret,mtx,dist,rvecs,tvecs,corners,h,w= get_inner_mtx(photo_path,w1,h1)
        c2w = calibration_photo(rvecs[-1],tvecs[-1],w1,h1)
        # c2w = calibration(ret,mtx,dist,rvecs[-1],tvecs[-1],corners,w1,h1)
        c2w_metrix.append(c2w)


    os.makedirs(os.path.join(os.getcwd(),'log'),exist_ok=True)
    f = os.path.join(os.getcwd(),'log','c2w_metrix.pkl')
    log = {'c2w_metrix':c2w_metrix,'ret':ret,'rvecs':rvecs,'h:':h,'w:':w}
    return c2w_metrix,h,w

