# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:18:42 2019

@author: mingjay
"""

from sklearn import datasets, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split



## 讀取鳶尾花資料集
#iris = datasets.load_iris()
#
## 切分訓練集/測試集
#x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)
#
## 建立模型
#clf = GradientBoostingClassifier()
#
## 訓練模型
#clf.fit(x_train, y_train)
#
## 預測測試集
#y_pred = clf.predict(x_test)
#
#acc = metrics.accuracy_score(y_test, y_pred)
#print("Acuuracy: ", acc)
#
#
digits = datasets.load_digits()


# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=4)

# 建立模型
clf = GradientBoostingClassifier()

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)
#print(digits.feature_names)
print("Feature importance: ", clf.feature_importances_)