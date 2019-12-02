# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:14:20 2019

@author: mingjay
"""

import numpy as np 
# Sigmoid 函數可以將任何值都映射到一個位於 0 到  1 範圍內的值。通過它，我們可以將實數轉化為概率值
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])  
        
# define y for output dataset            
y = np.array([[0,0,1,1]]).T
print(y.shape)

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
#亂數設定產生種子得到的權重初始化集仍是隨機分佈的，
#但每次開始訓練時，得到的權重初始集分佈都是完全一致的。
 
# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
# define syn1

iter = 0
#該神經網路權重矩陣的初始化操作。
#用 “syn0” 來代指 (即“輸入層-第一層隱層”間權重矩陣）
#用 “syn1” 來代指 (即“輸入層-第二層隱層”間權重矩陣）
syn1_history = [syn1]
syn0_history = [syn0]

print(syn0, syn1)

error_history = np.zeros(shape=(10000))
for iter in range(10000):
    # forward propagation
    
    l0 = X
    l1 = nonlin(l0.dot(syn0))
    l2 = nonlin(l1.dot(syn1))

    '''
    新增
    l2_error 該值說明了神經網路預測時“丟失”的數目。
    l2_delta 該值為經確信度加權後的神經網路的誤差，除了確信誤差很小時，它近似等於預測誤差。
    '''
 
    # how much did we miss?

    error = 1/2*(np.sum(y - l2)**2)
    error_history[iter] = error
    
    l2_delta = (y-l2) * nonlin(l2, deriv=True)
    
    
    l1_delta = syn1.T.dot(l2_delta)* nonlin(l1, deriv=True) 
    #l1_delta = (l2-l1)* nonlin(l1, deriv=True) 
    # update weights
    syn0 -= np.dot(l0.T,l1_delta)
    syn1 -= np.dot(l1.T,l2_delta)
     # syn1 update weights
    
print("Output After Training:")


import matplotlib.pyplot as plt
#%matplotlib inline 
#適用於 Jupyter Notebook, 宣告直接在cell 內印出執行結果

plt.plot(syn0_history[0], ms=3, lw=1.5, color='black')
plt.xlabel(r'$L1$', fontsize=16)
plt.show()

plt.plot(syn1_history[0], ms=3, lw=1.5, color='black')
plt.xlabel(r'$L2$', fontsize=16)
plt.show()


plt.plot(range(10000), error_history)
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.show()