from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from package.SimpleLinearRegression2 import SimpleLinearRegression2
from package.train_test_split import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 加载波士顿房价数据集
boston = datasets.load_boston()

# 打印数据集说明
print(boston.DESCR)
# .. _boston_dataset:

# Boston house prices dataset
# ---------------------------

# **Data Set Characteristics:**

#     :Number of Instances: 506

#     :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

#     :Attribute Information (in order):
#         - CRIM     per capita crime rate by town
#         - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#         - INDUS    proportion of non-retail business acres per town
#         - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#         - NOX      nitric oxides concentration (parts per 10 million)
#         - RM       average number of rooms per dwelling # 每个住宅平均房间数
#         - AGE      proportion of owner-occupied units built prior to 1940
#         - DIS      weighted distances to five Boston employment centres
#         - RAD      index of accessibility to radial highways
#         - TAX      full-value property-tax rate per $10,000
#         - PTRATIO  pupil-teacher ratio by town
#         - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
#         - LSTAT    % lower status of the population
# ...
#    - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
# # 数据集特征名
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = boston.data[:, 5]  # 取出所有行和第五列，只使用房间数量这个特征
print(x.shape)  # (506,) x 是有506行的一个向量

y = boston.target
print(y.shape)  # (506,) x 是有506行的一个向量

# 绘制x,y的散点图
plt.scatter(x, y)
plt.show()

# 数据中大于最大值的被记录为最大值，不符合线性回归特征，应该删除
max = np.max(y)
x = x[y < max]  # [y < max] 是返回了一个布尔值作向量的索引
y = y[y < max]
plt.scatter(x, y)
plt.show()


# 使用简单线性回归法
x_train, x_test, y_train, y_test = train_test_split(x, y, seed=666)
reg = SimpleLinearRegression2()
reg.fit(x_train, y_train)
plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color="r")
plt.show()

y_predict = reg.predict(x_test)

# MSE
mse_test = np.sum((y_predict - y_test) ** 2) / len(x_test)
print(mse_test)  # 24.156602134387438

# RMSE
rmse_test = sqrt(mse_test)
print(rmse_test)  # 4.914936635846635

# MAE
mae_test = np.sum(np.absolute(y_predict - y_test)) / len(x_test)
print(mae_test)  # 3.5430974409463873

# 计算 R squared
print(1 - mean_squared_error(y_test, y_predict) / np.var(y_test))

# 使用scikit-learn 封装的MSE和MAE
print(mean_squared_error(y_test, y_predict))  # 24.156602134387438
print(mean_absolute_error(y_test, y_predict))  # 3.5430974409463873
print(r2_score(y_test, y_predict))
