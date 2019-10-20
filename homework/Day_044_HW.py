# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:57:21 2019

@author: mingjay
"""

from sklearn import datasets, metrics 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# 如果是分類問題，請使用 DecisionTreeClassifier，若為回歸問題，請使用 DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型 (使用 20 顆樹，每棵樹的最大深度為 4)
clf = RandomForestClassifier(n_estimators=20, max_depth=4)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

#print(iris.feature_names)

#print("Feature importance: ", clf.feature_importances_)

#HW1==========================================================================
#試著調整 RandomForestClassifier(...) 中的參數，並觀察是否會改變結果？
# 建立模型

clf = RandomForestClassifier(n_estimators=5, max_depth=4)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy(n_estimator = 5): ", acc)

#print(iris.feature_names)

#print("Feature importance(n_estimator = 10): ", clf.feature_importances_)

#試著調整 RandomForestClassifier(...) 中的參數，並觀察是否會改變結果？
# 建立模型

clf = RandomForestClassifier(n_estimators=20, max_depth=1)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy(max_depth = 1): ", acc)

#print(iris.feature_names)

#print("Feature importance(max_depth = 1): ", clf.feature_importances_)

#HW2==========================================================================
#改用其他資料集 (boston, wine)，並與回歸模型與決策樹的結果進行比較

# 讀取資料集
#Consider feature1
wine=datasets.load_wine()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.1, random_state=4)

# 建立模型
clf = DecisionTreeClassifier()

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy(DecisionTree(wine)): ", acc)

#print(wine.feature_names)

#print("Feature importance(DecisionTree): ", clf.feature_importances_)
#===============================================================================

clf = RandomForestClassifier(n_estimators=10, max_depth=4)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy(RandomForest(wine)): ", acc)

#print(iris.feature_names)

#print("Feature importance(RandomForest): ", clf.feature_importances_)

