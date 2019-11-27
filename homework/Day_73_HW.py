# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 21:38:52 2019

@author: mingjay
"""

def LR_vs_iters(lr):

    cur_x = 3 # The algorithm starts at x=3
    #lr = 0.05 # Learning rate
    precision = 0.000001 #This tells us when to stop the algorithm
    previous_step_size = 1 #
    max_iters = 10000 # maximum number of iterations
    iters = 0 #iteration counter
    df = lambda x: 2*(x+5) #Gradient of our function 
    
    iters_history = [iters]
    x_history = [cur_x]


    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x #Store current x value in prev_x
        cur_x = cur_x - lr * df(prev_x) #Gradient descent
        previous_step_size = abs(cur_x - prev_x) # 取較大的值, Change in x
        iters = iters+1 #iteration count
        print("Iteration",iters,"\nX value is",cur_x) #Print iterations
        # Store parameters for plotting
        iters_history.append(iters)
        x_history.append(cur_x)


    print("Totally iteations: ", iters)
    print("The local minimum occurs at", cur_x)

    import matplotlib.pyplot as plt
    
    #適用於 Jupyter Notebook, 宣告直接在cell 內印出執行結果
    
    plt.plot(iters_history, x_history, 'o-', ms=3, lw=1.5, color='black')
    plt.xlabel(r'$iters$', fontsize=16)
    plt.ylabel(r'$x$', fontsize=16)
    plt.show()
    
    return iters

lr = [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]

I = []

for i in range(len(lr)):
    iters = LR_vs_iters(lr[i])
    I.append(iters)
    
plt.semilogx(lr,I)
plt.xlabel('lr')
plt.ylabel('iters')
plt.title('LR vs Iters on searching the min. of function y=(x+5)²')