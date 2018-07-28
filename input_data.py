# coding:utf-8

import os.path
import sys
import re
import os
import json

import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import pickle as pickle #python pkl 文件读写

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MyData():
    def __init__(self):
        self.data_filePath = []
        self.data_fileName = []

        self.data = []
        self.labels = []

# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    data = MyData()
    for allDir in pathDir:
        if allDir != ".DS_Store":
            child = os.path.join('%s/%s' % (filepath, allDir))
            if os.path.isfile(child):
                data.data_filePath.append(child)
                data.data_fileName.append(allDir)
                theTpye = re.split('-',allDir)[0]
                # print(theTpye)
                data.labels.append( int(theTpye)-1 )
    # # 显示
    # for i in array:
    #     print(i)      
    return data


def myFastGFile(py_data):
    # 新建一个Session
    with tf.Session() as sess:
        '''
        image_raw_data = tf.gfile.FastGFile(py_data.data_filePath[0], 'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data)
        plt.imshow(img_data.eval())
        plt.show()

        resized = tf.image.resize_images(img_data, [28, 28], method=0)
        print(resized)
        resized = tf.reshape(resized, [28, 28, 3]) #最后一维代表通道数目，如果是rgb则为3 
        print(resized)
        # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
        print("Digital type: ", resized.dtype)
        resized = np.asarray(resized.eval(), dtype='uint8')
        
        # tf.image.convert_image_dtype(rgb_image, tf.float32)
        plt.imshow(resized)
        plt.show()
        '''
        # path = py_data.data_filePath[0]
        for path in py_data.data_filePath:
            # 读取文件
            image_raw_data = tf.gfile.FastGFile(path, 'rb').read()
            # 解码
            img_data = tf.image.decode_jpeg(image_raw_data)
            # print(img_data)
            # 转灰度图
            # img_data = sess.run(tf.image.rgb_to_grayscale(img_data))  
            # 改变图片尺寸
            resized = tf.image.resize_images(img_data, [100, 100], method=0)
            # 设定 shape
            # resized = tf.reshape(resized, [28, 28, 1]) #最后一维代表通道数目，如果是rgb则为3 
            resized = tf.reshape(resized, [100, 100, 3]) #最后一维代表通道数目，如果是rgb则为3 
            # 标准化
            # standardization_image = resized
            standardization_image = tf.image.per_image_standardization(resized)#标准化
            # print(standardization_image)
            # print(standardization_image.eval())
            resized = tf.reshape(standardization_image, [-1]) #最后一维代表通道数目，如果是rgb则为3 
            # resized = tf.reshape(resized, [-1]) #最后一维代表通道数目，如果是rgb则为3 

            ## 链接     
            ## resized = tf.expand_dims(resized, 0) # 增加一个维度
            ## print(resized)
            ## print(py_data.data)
            ## test_data = tf.concat(0, [test_data, resized])
            
            py_data.data.append(resized.eval())

        '''
        # #验证数据转换正确
        resized = tf.reshape(py_data.data[0], [100, 100, 3])
        resized = np.asarray(resized.eval(), dtype='uint8')        
        plt.imshow(resized)
        plt.show()
        '''
        
def saveData(py_data, filePath_data, filePath_labels):
    pass
    '''
    with tf.Session() as sess:
        train_data =tf.convert_to_tensor(np.array( trainData.data ) )
    '''
    data = np.array( py_data.data )
    labels = py_data.labels

    # import os
    if os.path.exists(filePath_data): #删除文件，可使用以下两种方法。
        os.remove(filePath_data)      #os.unlink(my_file)

    if os.path.exists(filePath_labels): #删除文件，可使用以下两种方法。
        os.remove(filePath_labels)      #os.unlink(my_file)


    with open(filePath_data,'wb') as f:
        pickle.dump(data, f)

    with open(filePath_labels,'wb') as f:
        pickle.dump(labels, f)

    print('\ndone!')

def run(dataPath, plkFileName, plkFileNameLabels):
    # dataPath = "data/train"
    # plkFileName = "train_data.plk"
    # plkFileNameLabels = "train_labels.plk"

    # 遍历每一个文件
    loadData = eachFile(dataPath) #注意：末尾不加/
    # 转换类型
    myFastGFile(loadData)  
    # 保存转换后的数据
    saveData(loadData, plkFileName, plkFileNameLabels)


if __name__ == "__main__":
    pass
    print('目前系统的编码为：',sys.getdefaultencoding()) 

    # ## 训练集 - 60%
    # run("data/train", "cache/train_data.plk", "cache/train_labels.plk")

    # ## 评估集 - 20%
    # run("data/valid", "cache/valid_data.plk", "cache/valid_labels.plk")

    # ## 测试集 - 20%
    # run("data/test", "cache/test_data.plk", "cache/test_labels.plk")
    
    ## 测试集 - 20%
    run("data/test_temporary", "cache/test_temporary_data.plk", "cache/test_temporary_labels.plk")


