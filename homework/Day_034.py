# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 20:21:55 2019

@author: mingjay
"""

from sklearn.model_selection import train_test_split, KFold
import numpy as np

X = np.arange(50).reshape(10, 5) # 生成從 0 到 50 的 array，並 reshape 成 (10, 5) 的 matrix
y = np.zeros(10) # 生成一個全零 arrary
y[:5] = 1 # 將一半的值改為 1
print("Shape of X: ", X.shape)
print("Shape of y: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

kf = KFold(n_splits=5)
i = 0
for train_index, test_index in kf.split(X):
    i +=1 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("FOLD {}: ".format(i))
    print("X_test: ", X_test)
    print("Y_test: ", y_test)
    print("-"*30)
    
#HW
    
import numpy as np
X = np.arange(1000).reshape(200, 5)
y = np.zeros(200)
y[:40] = 1
    
X_train0, X_test0, y_train0, y_test0 = train_test_split(X[40:], y[40:], test_size=(10/160), random_state=42) 

X_train1, X_test1, y_train1, y_test1 = train_test_split(X[:40], y[:40], test_size=(10/40), random_state=42) 

y_test = np.concatenate((y_test0, y_test1))
