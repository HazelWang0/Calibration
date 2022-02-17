import cv2
import numpy as np
import os
import sys
import glob
import pickle
# import math
from .utils import arse_config
from .trans import read_pic, read_pic

import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    dots = np.array(cp).astype(int) # list,在画图前必须转为int类型
    print(type(cp))
    print(dots[0],dots[1],sep=';')
    print(dots.shape[0])
    for i in range(dots.shape[0]):
        dot = dots[i]
        print('dot:',dot)
        print('i:',i)

        ax.scatter(dot[0],dot[1],dot[2],c='g', marker='o')
        ax.text(dot[0][0],dot[1][0],dot[2][0],i,fontsize=12, color = "r", style = "italic")
    # --------------------------------  --------------------------------
    # 设置坐标轴标题和刻度
    ax.set(xlabel='X',
        ylabel='Y',
        zlabel='Z',
        xlim=(0, 9),
        ylim=(0, 9),
        zlim=(0, 9),
        xticks=np.arange(-400, 400, 80),
        yticks=np.arange(-400, 400, 80),
        zticks=np.arange(-800, 800, 160)
        )

    # 调整视角
    ax.view_init(elev=15,    # 仰角
                azim=60   # 方位角
                )

    # 显示图形
    plt.show()

def showsurface(c2w,cp):
    fig = plt.figure(figsize=(5, 5),
                    facecolor='whitesmoke'
                    )
    # 创建 3D 坐标系
    ax = fig.gca(fc='whitesmoke',
                projection='3d' 
                )

    # x1 = np.linspace(0,2,40)
    # y1 = np.linspace(0,2,40)
    x = c2w[0]
    X, Y = np.meshgrid(x, y)
    z1 =X*2,
    
    ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='rainbow')
    
    plt.show()
 

def draw(img, corners, imgpts):
    # img  = np.trunc(10*img)
    img = img.astype(int)
    img = np.mat(img)
    corners = corners.astype(int)
    imgpts = imgpts.astype(int)
    print(type(corners))
    print(type(imgpts))
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img




# 获得像素坐标
def calibration_photo(photo_path,mtx,dist):
    # 设置要标定的角点个数
    x_nums = 11  # x方向上的角点个数
    y_nums = 8
    # 设置(生成)标定图在世界坐标中的坐标
    world_point = np.zeros((x_nums * y_nums, 3), np.float32)  # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素
    world_point[:, :2] = 15 * np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)  # mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行
    # print('world point:',world_point)
    # .T矩阵的转置
    # reshape()重新规划矩阵，但不改变矩阵元素
    # 设置世界坐标的坐标
    axis = 15* np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image = cv2.imread(photo_path)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 查找角点
    ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums), )
    # print(ok)
    if ok:
        # 获取更精确的角点位置
        exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 获取外参
        _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)
        #获得的旋转矩阵是向量，是3×1的矩阵，想要还原回3×3的矩阵，需要罗德里格斯变换Rodrigues，
        
        rotation_m, _ = cv2.Rodrigues(rvec)#罗德里格斯变换
        print(rotation_m)
        print('旋转矩阵是：\n', rvec)
        print('平移矩阵是:\n', tvec)
        rotation_t = np.hstack([rotation_m,tvec])
        rotation_t_Homogeneous_matrix = np.vstack([rotation_t,np.array([[0, 0, 0, 1]])])
        print('w2c',rotation_t_Homogeneous_matrix)
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        # # 可视化角点
        # img = draw(image, corners, imgpts)
        # cv2.imshow('img', img)
        cameraPosition = -np.matrix(rotation_m).T * np.matrix(tvec)
        print('cp',cameraPosition)
        return rotation_t_Homogeneous_matrix,cameraPosition # 返回旋转矩阵和平移矩阵组成的齐次矩阵



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

    print('mtx:',mtx)
    focal = mtx[0:1,0:1]
    print('focal:',focal)
    return(ret,mtx,dist,rvecs,tvecs,corners,focal)


def set_calibration():
    
    args = arse_config()
    read_pic(args.folder)
    root = os.path.join(args.folder,'*.png') # 标定图像保存路径
    photos_path = glob.glob(root)
    w1 = args.config[0]
    h1 = args.config[1] 
    c2w_metrix = []
    cp_mertix = []
    print('calibrating')
    for photo_path in photos_path:
        # ret相机？？ mtx相机内参，dist相机畸变,rvecs旋转向量，tvecs平移向量,focal焦距
        ret,mtx,dist,rvecs,tvecs,corners,focal = get_inner_mtx(photo_path,w1,h1)        
        c2w,cp = calibration_photo(photo_path,mtx,dist)
        c2w_metrix.append(c2w)
        cp_mertix.append(cp)
    # show(c2w_metrix,cp_mertix)
        



    # os.makedirs(os.path.join(os.getcwd(),'log'),exist_ok=True)
    # f = os.path.join(os.getcwd(),'log','c2w_metrix.pkl')
    # log = {'c2w_metrix':c2w_metrix,'ret':ret,'rvecs':rvecs}
    # with open(f, 'wb') as file:
    #     pickle.dump(log, file)
    # print('finish calibrating')

