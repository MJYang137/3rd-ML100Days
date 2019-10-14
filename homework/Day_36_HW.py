# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:47:25 2019

@author: mingjay
"""


from sklearn import metrics
import numpy as np
y_pred = np.random.randint(2, size=100)  # 生成 100 個隨機的 0 / 1 prediction
y_true = np.random.randint(2, size=100)  # 生成 100 個隨機的 0 / 1 ground truth

y_pred

def f_score(beta,precision,recall):
    return (1+beta**2)*(precision*recall)/(beta**2*precision+recall)

precision = metrics.precision_score(y_true,y_pred)
recall = metrics.recall_score(y_true,y_pred)

beta = 2

f2 = f_score(beta,precision,recall)
