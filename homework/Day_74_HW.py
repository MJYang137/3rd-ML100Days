# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:53:09 2019

@author: mingjay
"""

import numpy as np
import matplotlib.pyplot as plt

#梯度下降法
# 目標函數:y=(x+3)^2
def func(x): 
    return np.square(x+3)

# 目標函數一階導數:dy/dx=2*(x+3)
def dfunc(x): 
    return 2 * (x+3)

def GD(w_init, df, epochs, lr):    
    """  梯度下降法。給定起始點與目標函數的一階導函數，求在epochs次反覆運算中x的更新值
        :param w_init: w的init value    
        :param df: 目標函數的一階導函數    
        :param epochs: 反覆運算週期    
        :param lr: 學習率    
        :return: x在每次反覆運算後的位置   
     """    
    xs = np.zeros(epochs+1) # 把 "epochs+1" 轉成dtype=np.float32    
    x = w_init    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # v表示x要跨出的幅度        
        v = - dx * lr        
        x += v        
        xs[i+1] = x    
    return xs

# 起始權重
w_init = 3    
# 執行週期數
epochs = 20 
# 學習率   
#lr = 0.3
lr = 0.01
# 梯度下降法 
x = GD(w_init, dfunc, epochs, lr=lr) 
print (x)

#劃出曲線圖
color = 'r'    
 
from numpy import arange
t = arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(x, func(x), c=color, label='lr={}'.format(lr))    
plt.scatter(x, func(x), c=color, )    
plt.legend()

plt.show()

#學習率對梯度下降法的影響
#學習率較小時，收斂到正確結果的速度較慢。 學習率較大時，容易在搜索過程中發生震盪。

line_x = np.linspace(-5, 5, 100)
line_y = func(line_x)
plt.figure('Gradient Desent: Learning Rate')

w_init = 3
epochs = 5
x = w_init
lr = [0.0001, 0.01,0.1,0.5]

color = ['r', 'g', 'y','b']
#size = np.ones(epochs+1) * 10
#size[-1] = 70
for i in range(len(lr)):
    x = GD(w_init, dfunc, epochs, lr=lr[i])
    plt.subplot(1, 4, i+1)
    plt.plot(line_x, line_y, c='b')
    plt.plot(x, func(x), c=color[i], label='lr={}'.format(lr[i]))
    plt.scatter(x, func(x), c=color[i])
    plt.legend()
plt.show()


#學習率衰減公式

def GD_decay(w_init, df, epochs, lr, decay):
    xs = np.zeros(epochs+1)
    x = w_init
    xs[0] = x
    v = 0
    for i in range(epochs):
        dx = df(x)
        # 學習率衰減 
        lr_i = lr * 1.0 / (1.0 + decay * i)
        # v表示x要改变的幅度
        v = - dx * lr_i
        x += v
        xs[i+1] = x
    return xs


line_x = np.linspace(-5, 5, 100)
line_y = func(line_x)
plt.figure('Gradient Desent: Decay')

lr = 1.4
iterations = np.arange(300)
decay = [0.0, 0.01, 0.4, 0.6, 0.8, 0.99]
for i in range(len(decay)):
    decay_lr = lr * (1.0 / (1.0 + decay[i] * iterations))
    plt.plot(iterations, decay_lr, label='decay={}'.format(decay[i]))

plt.ylim([0, 1.1])
plt.legend(loc='best')
plt.show()

#Momentum (動量)
line_x = np.linspace(-10, 5, 100)
line_y = func(line_x)
plt.figure('Gradient Desent: Decay')

x_start = 3
epochs = 100

lr = [0.1, 0.3, 0.9, 0.99]
decay = [0.0, 0.01, 0.5, 0.9]

color = ['k', 'r', 'g', 'y']

row = len(lr)
col = len(decay)
size = np.ones(epochs + 1) * 10
size[-1] = 70
plt.figure(figsize = (20,10))
for i in range(row):
     for j in range(col):
        x = GD_decay(x_start, dfunc, epochs, lr=lr[i], decay=decay[j])
        plt.subplot(row, col, i * col + j + 1)
        plt.plot(line_x, line_y, c='b')
        plt.plot(x, func(x), c=color[i], label='lr={}, de={}'.format(lr[i], decay[j]))
        plt.scatter(x, func(x), c=color[i], s=size)
        plt.legend(loc=0)
plt.show()