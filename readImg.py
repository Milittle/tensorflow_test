# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:07:05 2017

@author: milittle
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

person = [('%03d' % x) for x in range(1, 125)]

not_in_person = ['028', '034', '046', '048', '053', '064', '065', '067', '068', '074', '076', '077', '081', '098', '109', '116']

bg = ['bg-01', 'bg-02']
cl = ['cl-01', 'cl-02']
nm = ['nm-%02d' % x for x in range(1, 7)]
v = [x for x in range(181)]
view = ['%03d' % x for x in v[0:181:18]]

test_gei_path = 'D:\\data\\GEI\\test\\';

GEI_name = "GEI.png"

def readNM():
    j = 0
    GEI_NM = np.zeros((66 * 124, 210, 70), dtype = np.float32)
    for i in person:
        if i in not_in_person:
            continue
        for n in nm:
            for vi in view:
                GEI = readImage(test_gei_path + os.sep + i + os.sep + n + os.sep + str(vi) + os.sep + GEI_name)
                GEI_NM[j] = GEI
                j += 1
    return GEI_NM

def readBG():
    j = 0
    GEI_BG = np.zeros((22 * 124, 210, 70), dtype = np.float32)
    for i in person:
        for n in bg:
            for vi in view:
                GEI = readImage(test_gei_path + os.sep + i + os.sep + n + os.sep + str(vi) + os.sep + GEI_name)
                GEI_BG[j] = GEI
                j += 1
    return GEI_BG
                
def readCL():
    j = 0
    GEI_CL = np.zeros((22 * 124, 210, 70), dtype = np.float32)
    for i in person:
        for n in cl:
            for vi in view:
                GEI = readImage(test_gei_path + os.sep + i + os.sep + n + os.sep + str(vi) + os.sep + GEI_name)
                GEI_CL[j] = GEI
                j += 1
    return GEI_CL

def readNM_(normaltype = 0): #normaltype = 0 represent nm-01
    GEI_NM_ = np.zeros((11 * 124, 210, 70), dtype = np.float32)
    j = 0
    for i in person:
        if i in not_in_person:
            continue
        for n in nm[normaltype:normaltype + 1]:
             for vi in view:
                GEI = readImage(test_gei_path + os.sep + i + os.sep + n + os.sep + str(vi) + os.sep + GEI_name)
                GEI_NM_[j] = GEI
                j += 1
    return GEI_NM_

def readImage(path):
    GEI = np.array(Image.open(path))
    return GEI

def readYLabel(path, num):
    yLabel = np.zeros([num,2])
    j = 0
    with open(path) as f:
        for i in f.readlines():
            strT = i.strip()
            fil = filter(None, strT.split(" "))
            yLabel[j] = [j for j in fil]
            j += 1
    return yLabel

batch_size = 0
batch_test_size = 0

def nextTB():
    data = readNM_()
    
    data_galley_ = list(data[709:1188:11, :, :].reshape(44, 14700))
    data_probe_ = list(data[710:1188:11, :, :].reshape(44, 14700))
    return data_galley_, data_probe_

def nextTestBatch():
    global batch_test_size
    if batch_test_size == 2310:
        batch_test_size = 0
    data = readNM_()

    
    data_list_1 = list(data[507:705:11, :, :].reshape(18, 14700))
    data_list_2 = list(data[507:705:11, :, :].reshape(18, 14700))
    
    data_list_1.extend(list(data[507:705:11, :, :].reshape(18, 14700)[0:17, :]))
    data_list_2.extend(list(data[507:705:11, :, :].reshape(18, 14700)[1:18, :]))
    
    for i in range(10):
        data_list_1.extend(list(data[507:705:11, :, :].reshape(18, 14700)))
        data_list_2.extend(list(data[i + 1 + 507:705:11, :, :].reshape(18, 14700)))
        data_list_1.extend(list(data[507:705:11, :, :].reshape(18, 14700)[0:17, :]))
        data_list_2.extend(list(data[i + 1 + 507:705:11, :, :].reshape(18, 14700)[1:18, :]))
    
    for i in range(10):
        for j in range(i, 10):
            data_list_1.extend(list(data[i + 1 + 507:705:11, :, :].reshape(18, 14700)))
            data_list_2.extend(list(data[j + 1 + 507:705:11, :, :].reshape(18, 14700)))
            data_list_1.extend(list(data[i + 1 + 507:705:11, :, :].reshape(18, 14700)[0:17, :]))
            data_list_2.extend(list(data[j + 1 + 507:705:11, :, :].reshape(18, 14700)[1:18, :]))

    data_ys = readYLabel('D:\\data\\GEI\\yTestLabel.txt', num = 2310)
    batch_test_size = batch_test_size + 128
    if batch_test_size > 2310:
        batch_test_size = 2310
    return data_list_1[batch_test_size - 128:batch_test_size], data_list_2[batch_test_size - 128:batch_test_size], data_ys[batch_test_size- 128:batch_test_size]
    #return data_list_1, data_list_2, data_ys
def nextBatch():
    global batch_size
    if batch_size == 6006:
        batch_size = 0
    data = readNM_()
            
    
#    data_0_ = data[0:506:11, :, :].reshape(46, 14700);
#    data_18_ = data[1:506:11, :, :].reshape(46, 14700);
#    data_36_ = data[2:506:11, :, :].reshape(46, 14700);
#    data_54_ = data[3:506:11, :, :].reshape(46, 14700);
#    data_72_ = data[4:506:11, :, :].reshape(46, 14700);
#    data_90_ = data[5:506:11, :, :].reshape(46, 14700);
#    data_108_ = data[6:506:11, :, :].reshape(46, 14700);
#    data_126_ = data[7:506:11, :, :].reshape(46, 14700);
#    data_144_ = data[8:506:11, :, :].reshape(46, 14700);
#    data_162_ = data[9:506:11, :, :].reshape(46, 14700);
#    data_180_ = data[10:506:11, :, :].reshape(46, 14700);
    
    data_list_1 = list(data[0:506:11, :, :].reshape(46, 14700))
    data_list_2 = list(data[0:506:11, :, :].reshape(46, 14700))
    
    data_list_1.extend(list(data[0:506:11, :, :].reshape(46, 14700)[0:45, :]))
    data_list_2.extend(list(data[0:506:11, :, :].reshape(46, 14700)[1:46, :]))
    
    for i in range(10):
        data_list_1.extend(list(data[0:506:11, :, :].reshape(46, 14700)))
        data_list_2.extend(list(data[i + 1:506:11, :, :].reshape(46, 14700)))
        data_list_1.extend(list(data[0:506:11, :, :].reshape(46, 14700)[0:45, :]))
        data_list_2.extend(list(data[i + 1:506:11, :, :].reshape(46, 14700)[1:46, :]))
    
    for i in range(10):
        for j in range(i, 10):
            data_list_1.extend(list(data[i + 1:506:11, :, :].reshape(46, 14700)))
            data_list_2.extend(list(data[j + 1:506:11, :, :].reshape(46, 14700)))
            data_list_1.extend(list(data[i + 1:506:11, :, :].reshape(46, 14700)[0:45, :]))
            data_list_2.extend(list(data[j + 1:506:11, :, :].reshape(46, 14700)[1:46, :]))
    
#    data_list_1 = data_0_[0:46, :]
#    data_list_1 = list(data_list_1)
#    data_list_1.extend(list(data_18_[0:46, :]))
#    data_list_1.extend(list(data_36_[0:46, :]))
#    data_list_1.extend(list(data_54_[0:46, :]))
#    data_list_1.extend(list(data_72_[0:46, :]))
#    data_list_1.extend(list(data_90_[0:46, :]))
#    data_list_1.extend(list(data_108_[0:46, :]))
#    data_list_1.extend(list(data_126_[0:46, :]))
#    data_list_1.extend(list(data_144_[0:46, :]))
#    data_list_1.extend(list(data_162_[0:46, :]))
#    data_list_1.extend(list(data_162_[0:46, :]))
#    
#    data_list_2 = data_0_[0:46, :]
#    data_list_2 = list(data_list_2)
#    data_list_2.extend(list(data_18_[0:46, :]))
#    data_list_2.extend(list(data_36_[0:46, :]))
#    data_list_2.extend(list(data_54_[0:46, :]))
#    data_list_2.extend(list(data_72_[0:46, :]))
#    data_list_2.extend(list(data_90_[0:46, :]))
#    data_list_2.extend(list(data_108_[0:46, :]))
#    data_list_2.extend(list(data_126_[0:46, :]))
#    data_list_2.extend(list(data_144_[0:46, :]))
#    data_list_2.extend(list(data_162_[0:46, :]))
#    data_list_2.extend(list(data_180_[0:46, :]))
#    
#    data_list_1_ = list(data_0_[0:45, :])
#    data_list_1_.extend(list(data_18_[0:45, :]))
#    data_list_1_.extend(list(data_36_[0:45, :]))
#    data_list_1_.extend(list(data_54_[0:45, :]))
#    data_list_1_.extend(list(data_72_[0:45, :]))
#    data_list_1_.extend(list(data_90_[0:45, :]))
#    data_list_1_.extend(list(data_108_[0:45, :]))
#    data_list_1_.extend(list(data_126_[0:45, :]))
#    data_list_1_.extend(list(data_144_[0:45, :]))
#    data_list_1_.extend(list(data_162_[0:45, :]))
#    
#    data_list_2_ = data_18_[1:46, :]
#    data_list_2_ = list(data_list_2_)
#    data_list_2_.extend(list(data_36_[1:46, :]))
#    data_list_2_.extend(list(data_54_[1:46, :]))
#    data_list_2_.extend(list(data_72_[1:46, :]))
#    data_list_2_.extend(list(data_90_[1:46, :]))
#    data_list_2_.extend(list(data_108_[1:46, :]))
#    data_list_2_.extend(list(data_126_[1:46, :]))
#    data_list_2_.extend(list(data_144_[1:46, :]))
#    data_list_2_.extend(list(data_162_[1:46, :]))
#    data_list_2_.extend(list(data_180_[1:46, :]))
#    
#    data_list_1.extend(data_list_1_)
#    data_list_2.extend(data_list_2_)
    data_ys = readYLabel('D:\\data\\GEI\\yLabel.txt', num = 6006)
    batch_size = batch_size + 128
    if batch_size > 6006:
        batch_size = 6006
    return data_list_1[batch_size - 128:batch_size], data_list_2[batch_size - 128:batch_size], data_ys[batch_size- 128:batch_size]    

data1, data2, data_y = nextTestBatch()

data_galley,data_probe = nextTB()

nm1 = readNM_()

plt.imshow(nm1[0], cmap = 'gray')

plt.show()