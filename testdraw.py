import cv2
import numpy as np
import os
import sys
import glob
import pickle
import math

import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show():
    # 创建画布
    fig = plt.figure(figsize=(12, 8),
                    facecolor='lightyellow'
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
    # x = [1,2,3,4,5,6,7,8,9,10]
    # y = [5,6,7,8,2,5,6,3,7,2]
    # z = [1,2,6,3,2,7,3,3,7,2]

    # ax.scatter(x, y, z, c='g', marker='o')

    
def showquiver():
    # cp = [[1,2,3],[4,5,6]]
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    # x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
    #                     np.arange(-0.8, 1, 0.2),
    #                     np.arange(-0.8, 1, 0.8))
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                    np.arange(-0.8, 1, 0.2),
                    np.arange(-0.8, 1, 0.8))

    # Make the direction data for the arrows
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
        np.sin(np.pi * z))

    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
    plt.show()

def plot():
    cp = [[1,2,3],[4,5,6]]
    fig = plt.figure(figsize=(5, 5),
                    facecolor='white'
                    )
    # 创建 3D 坐标系
    ax = fig.gca(fc='white',
                projection='3d' 
                )

    # x1 = np.linspace(0,2,40)
    # y1 = np.linspace(0,2,40)
    x1,y1 = np.meshgrid()
    x1, y1 = np.meshgrid(x1, y1)
    z1 =x1**2+(2-y1)**2
    
    ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap='rainbow')
    
    plt.show()

def quiver():
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    # x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
    #                     np.arange(-0.8, 1, 0.2),
    #                     np.arange(-0.8, 1, 0.8))
    # x,y,z = np.meshgrid([1,2,3],[4,5,6],[7,8,9])
    # u = [1,2,3]
    # v = [4,5,6]
    # w = [7,8,9]
    l = np.array([[1],[4],[7]])
    x,y,z = np.meshgrid([l[0]],[l[1]],[l[2]])
    u = [1]
    v = [4]
    w = [7]

    print('x',x)
    # Make the direction data for the arrows
    # u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    # v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    # w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
    #     np.sin(np.pi * z))

    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

    plt.show()
    
def get():
    m = np.eye(3)
    focal = m[0:1,0:1]
    print(focal)


if __name__ == '__main__':
    quiver()