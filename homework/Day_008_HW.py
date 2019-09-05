# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:53:19 2019

@author: mingjay
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dir_data = 'C:\MJ_Python_codes\ML\Kaggle'

f_app_train = os.path.join(dir_data, 'application_train.csv')

app_train = pd.read_csv(f_app_train)


columnlist = list(app_train.columns)

df = app_train

# 秀出資料欄位的類型與數量
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df
#確定只有 int64, float64, object 三種類型後, 分別將欄位名稱存於三個 list 中
int_features = []
float_features = []
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64':
        float_features.append(feature)
    elif dtype == 'int64':
        int_features.append(feature)
    else:
        object_features.append(feature)

print(f'{len(float_features)} Float Features : {float_features}\n')


#
a = df[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]
a_sta = a.describe()

X = list(a.columns)
mean = [a_sta.iloc[1,0],a_sta.iloc[1,1],a_sta.iloc[1,2]]
std = [a_sta.iloc[2,0],a_sta.iloc[2,1],a_sta.iloc[2,2]]
fig1 =plt.figure()
plt.bar(X,mean)
plt.xlabel('Selected Columns')
plt.ylabel('Mean')
fig2 =plt.figure()
plt.bar(X,std)
plt.xlabel('Selected Columns')
plt.ylabel('Std. Dev.')
#
fig3 =plt.figure()
b = df['AMT_ANNUITY']
b.describe()
bb = np.array(list(b))
#bb = np.log1p(bb)

n,bins,patches = plt.hist(bb,bins=100,normed=False)
plt.xlabel('AMT_ANNUITY')
plt.ylabel('Population')
plt.title('')
