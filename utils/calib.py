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
def calibration_photo(inter_corner_shape, size_per_grid, img_dir,img_type):
        # criteria: only for subpix calibration, which is not used here.
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w,h = inter_corner_shape
    # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
    # like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w*h,3), np.float32)
    cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    # cp_world: corner point in world space, save the coordinate of corner points in world space.
    cp_world = cp_int*size_per_grid
    
    obj_points = [] # the points in world space
    img_points = [] # the points in image space (relevant to obj_points)
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        print("draw")
        # find the corners, cp_img: corner points in pixel space.
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w,h), None)
        # if ret is True, save.
        if ret == True:
            # cv2.cornerSubPix(gray_img,cp_img,(11,11),(-1,-1),criteria)
            obj_points.append(cp_world)
            img_points.append(cp_img)
            # view the corners
            cv2.drawChessboardCorners(img, (w,h), cp_img, ret)
            cv2.imshow('FoundCorners',img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    # calibrate the camera,cv2.calibrateCamera() renturn the calibrate results
    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)
    # uv is image.shape[:2]
        # calculate the error of reproject

    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_inter, coff_dis,(480,640), 0, (480,640))
    total_error = 0
    for i in range(len(obj_points)):
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2)/len(img_points_repro)
        total_error += error
    print(("Average Error of Reproject: "), total_error/len(obj_points))

    v_rot = np.array(v_rot)
    v_trans = np.array(v_trans)
    return v_rot,v_trans



def get_c2w(v_rot,v_trans):
    # print('shape v_rot:',v_rot[0].shape)
    rotation_m, _ = cv2.Rodrigues(v_rot) #罗德里格斯变换
    # print("rotation_matrix:",rotation_m)
    rotation_t = np.hstack([rotation_m,v_trans])
    rotation_t_Homogeneous_matrix = np.vstack([rotation_t,np.array([[0, 0, 0, 1]])])
    print('w2c:',rotation_t_Homogeneous_matrix)
    c2w = np.linalg.inv(rotation_t_Homogeneous_matrix)
    print('c2w:',c2w)


    v_rot = np.array(v_rot,dtype = float)
    v_trans = np.array(v_trans,dtype = float)
    
    return c2w


def get_pose(c2w,w1,h1,focal):
    poses = c2w[:, :3, :4].transpose([1,2,0])
    hwf = np.array([h1,w1,focal]).reshape([3,1])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    print('poses:',poses)
    return poses


def saveBlender(c2w_metrix,camera_angle_x_metrix,rvecs):
    os.makedirs(os.path.join(os.getcwd(),'log'),exist_ok=True)
    f = os.path.join(os.getcwd(),'log','c2w_metrix.pkl')
    log = {'c2w_metrix':c2w_metrix,'camera_angle_x':camera_angle_x_metrix,'rvecs':rvecs}
    with open(f, 'wb') as file:
        pickle.dump(log, file)
    print('finish calibrating')


def set_calibration():
    
    args = arse_config()
    read_pic(args.folder)
    root = args.folder # 标定图像保存路径
    size_per_grid = args.size_per_grid
    inter_corner_shape = args.inter_corner_shape
    photos_path = glob.glob(os.path.join(root,'*.png'))
    c2w_metrix = []
    camera_angle_x_metrix = []
    print('calibrating')
    v_rot,v_trans= calibration_photo(inter_corner_shape, size_per_grid, root,'png')
    print('shape rot:',v_rot.shape)
    print('trans rot:',v_trans.shape)

    print('photo_path',photos_path)
    print('len(photos_path)',len(photos_path))
    for i in range(len(photos_path)):
        print('i:',i)
        # ret相机？？ mtx相机内参，dist相机畸变,rvecs旋转向量，tvecs平移向量,focal焦距
        c2w = get_c2w(v_rot[i],v_trans[i])
        # pose = get_pose(c2w,cp,w1,h1,focal)
        c2w_metrix.append(c2w)
        # pose_metrix.append(pose)
    # showsurface(c2w_metrix,cp_mertix,rvecs,tvecs)
    
    print('camera_angle_x_metrix:',camera_angle_x_metrix)

    return c2w_metrix
    
    # saveBlender(c2w_metrix,camera_angle_x_metrix,rvecs)





