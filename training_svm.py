# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:01:00 2020

@author: WYZ
"""
from lib.file_operate import read_np_lists
import lib.SVM as svm
import numpy as np
import matplotlib.pyplot as plt

data_x = read_np_lists('HOG_data/PHOG.txt')
amount = len(data_x) 
print("containing "+str(amount)+" positive samples")
 
#train_x = data_x[0:int(amount*0.8)]
train_x = data_x[20:]
val_x = data_x[int(amount*0.8):amount]

train_y = np.array([1]*len(train_x))
val_y = np.array([1]*len(val_x))

data_nx = read_np_lists('HOG_data/NHOG.txt')
namount = len(data_nx) 
print("containing "+str(namount)+" negitive samples")
 
train_nx = data_nx[0:int(namount*0.5)]
val_nx = data_nx[int(namount*0.5):namount]

train_ny = np.array([-1]*len(train_nx))
val_ny = np.array([-1]*len(val_nx))

train_x = np.concatenate((train_x,train_nx))
val_x = np.concatenate((val_x,val_nx))

train_y = np.concatenate((train_y,train_ny))
val_y = np.concatenate((val_y,val_ny))

svm1=svm.svm_model(kernel='liner',C=0.05)
svm1.training(train_x,train_y)
right=0
for i in range(len(val_x)):
    result= svm1.Gsvm(val_x[i])
    #print(result)
    if result==val_y[i]:
        right+=1
    else:
        print(result,val_y[i])

svm1.save_model('trained_models/svm/svm_model.txt')
new_svm=svm.svm_model(load_path='trained_models/svm/svm_model.txt')
print('accuracy'+str(right/len(val_x)))
    