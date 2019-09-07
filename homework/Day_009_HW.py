# -*- coding: utf-8 -*-



#==============================================================================HW
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dir_data = 'C:\MJ_Python_codes\ML\Kaggle'

f_app_train = os.path.join(dir_data, 'application_train.csv')

app_train = pd.read_csv(f_app_train)
# 先篩選數值型的欄位
"""
YOUR CODE HERE, fill correct data types (for example str, float, int, ...)

"""
app_train.dtypes
#app_train.info()

#dtype_select = ['int64'] don't use this NOT RIGHT
dtype_select = [np.dtype('int64'),np.dtype('float64')]
numeric_columns = list(app_train.columns[list(app_train.dtypes.isin(dtype_select))])

# 再把只有 2 值 (通常是 0,1) 的欄位去掉
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])
print("Numbers of remain columns %s" % len(numeric_columns))

numeric_columns_selected = ['AMT_INCOME_TOTAL','REGION_POPULATION_RELATIVE','OBS_60_CNT_SOCIAL_CIRCLE']
# 檢視這些欄位的數值範圍
for col in numeric_columns_selected:
    """
    Your CODE HERE, make the box plot
    
    """
    plt.boxplot(app_train[col])
    plt.show()


# 最大值離平均與中位數很遠
print(app_train[numeric_columns_selected[0]].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
"""
YOUR CODE HERE
"""
cdf = app_train[numeric_columns_selected[0]].value_counts().sort_index().cumsum()


plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min(), cdf.index.max() * 1.05]) # 限制顯示圖片的範圍
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()

# 改變 y 軸的 Scale, 讓我們可以正常檢視 ECDF
plt.plot(np.log(list(cdf.index)), cdf/cdf.max())
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')

plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()

# 最大值落在分布之外
print(app_train[numeric_columns_selected[1]].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
"""
Your Code Here
"""
cdf = app_train[numeric_columns_selected[1]].value_counts().sort_index().cumsum()


plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.show()

app_train['REGION_POPULATION_RELATIVE'].hist()
plt.show()

app_train['REGION_POPULATION_RELATIVE'].value_counts()

# 就以這個欄位來說，雖然有資料掉在分布以外，也不算異常，僅代表這間公司在稍微熱鬧的地區有的據點較少，
# 導致 region population relative 在少的部分較為密集，但在大的部分較為疏漏

# 最大值落在分布之外
print(app_train[numeric_columns_selected[2]].describe())

# 繪製 Empirical Cumulative Density Plot (ECDF)
"""
Your Code Here
"""
cdf = app_train[numeric_columns_selected[2]].value_counts().sort_index().cumsum()


plt.plot(list(cdf.index), cdf/cdf.max())
plt.xlabel('Value')
plt.ylabel('ECDF')
plt.xlim([cdf.index.min() * 0.95, cdf.index.max() * 1.05])
plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍
plt.show()
# 改變 y 軸的 Scale, 讓我們可以正常檢視 ECDF
plt.plot(np.log(list(cdf.index)), cdf/cdf.max())
plt.xlabel('Value (log-scale)')
plt.ylabel('ECDF')

plt.ylim([-0.05,1.05]) # 限制顯示圖片的範圍

plt.show()
app_train['OBS_60_CNT_SOCIAL_CIRCLE'].hist()
plt.show()
print(app_train['OBS_60_CNT_SOCIAL_CIRCLE'].value_counts().sort_index(ascending = False))

# 把一些極端值暫時去掉，在繪製一次 Histogram
# 選擇 OBS_60_CNT_SOCIAL_CIRCLE 小於 20 的資料點繪製
"""
Your Code Here
"""
loc_a = app_train[app_train['OBS_60_CNT_SOCIAL_CIRCLE'] < 20]['OBS_60_CNT_SOCIAL_CIRCLE']
loc_b = 'OBS_60_CNT_SOCIAL_CIRCLE'

app_train.loc[loc_a, loc_b].hist()
plt.show()

#
loc_a = app_train[app_train['OBS_60_CNT_SOCIAL_CIRCLE'] < 50]['OBS_60_CNT_SOCIAL_CIRCLE']
bb = np.array(list(loc_a))
#bb = np.log1p(bb)

n,bins,patches = plt.hist(bb,bins=20,normed=False)
plt.yscale('log')
plt.xlabel('OBS_60_CNT_SOCIAL_CIRCLE')
plt.ylabel('Population')
plt.title('')