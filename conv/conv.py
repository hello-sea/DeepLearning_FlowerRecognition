import matplotlib.pyplot as plt
import pylab
import cv2 # pip install opencv-python
import numpy as np

''' 滤波-卷积核 '''
# 锐化
fil1 = np.array([ 
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
                ])
# Soble边缘检测 - 水平梯度
fil2 = np.array([ 
                [-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]
                ])
# Laplacian边缘检测
fil3 = np.array([
                [ 1, 1, 1],                        
                [ 1, 8, 1],
                [ 1, 1, 1]
                ])


def MyConv(filePath, fil):
    img = plt.imread(filePath)                        #在这里读取图片
    # plt.imshow(img)                                     #显示读取的图片
    # pylab.show()

    res = cv2.filter2D(img,-1,fil)                      #使用opencv的卷积函数
    # plt.imsave("conv/31-image_06924-res.jpg",res)

    # plt.imshow(res)                                     #显示卷积后的图片
    # pylab.show()
    return res

if __name__ == "__main__":
    filePath = "conv/31-image_06924.jpg"
    res = MyConv(filePath, fil1)
    plt.imsave("conv/31-image_06924-1.jpg",res)

    res = MyConv(filePath, fil2)
    plt.imsave("conv/31-image_06924-2.jpg",res)

    res = MyConv(filePath, fil3)
    plt.imsave("conv/31-image_06924-3.jpg",res)

    ##########################

    filePath = "conv/48-image_04666.jpg"
    res = MyConv(filePath, fil1)
    plt.imsave("conv/48-image_04666-1.jpg",res)

    res = MyConv(filePath, fil2)
    plt.imsave("conv/48-image_04666-2.jpg",res)

    res = MyConv(filePath, fil3)
    plt.imsave("conv/48-image_04666-3.jpg",res)
    
