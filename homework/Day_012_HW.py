# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:02:50 2019

@author: mingjay
"""

# 載入套件
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 讀取訓練與測試資料
data_path = 'C:\MJ_Python_codes\ML\Day_010_HW'
df_train = pd.read_csv(data_path + r'\titanic_train.csv')
df_test = pd.read_csv(data_path + r'\titanic_test.csv')

# 重組資料成為訓練 / 預測用格式
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
train_num = train_Y.shape[0] #train_Y.shape is a tuple, [0] to select element
df.head()

# 空值補 -1, 做羅吉斯迴歸
df_m1 = df.fillna(-1)
train_X = df_m1[:train_num]
estimator = LogisticRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print('空值補 -1, 做羅吉斯迴歸: %f'% cross_val_score(estimator, train_X, train_Y, cv=5).mean() )
"""
Your Code Here
"""
# 空值補 0, 做羅吉斯迴歸
df_0 = df.fillna(0)
train_X = df_0[:train_num]
estimator = LogisticRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print('空值補 0, 做羅吉斯迴歸: %f'% cross_val_score(estimator, train_X, train_Y, cv=5).mean() )
# 空值補平均值
df_mn = df.fillna(df.mean())
train_X = df_mn[:train_num]
estimator =  LogisticRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print('空值補平均值, 做羅吉斯迴歸: %f'% cross_val_score(estimator, train_X, train_Y, cv=5).mean() )

#使用不同的標準化方式 ( 原值 / 最小最大化 / 標準化 )，搭配羅吉斯迴歸模型，何者效果最好?

#原值
df_mn = df.fillna(df.mean())
train_X = df_mn[:train_num]
estimator = LogisticRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print('原值: %f'% cross_val_score(estimator, train_X, train_Y, cv=5).mean() )
# 最大最小化
df_temp = MinMaxScaler().fit_transform(df_mn)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print('最大最小化: %f'% cross_val_score(estimator, train_X, train_Y, cv=5).mean() )
# 標準化
df_temp = StandardScaler().fit_transform(df_mn)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print('標準化: %f'% cross_val_score(estimator, train_X, train_Y, cv=5).mean() )