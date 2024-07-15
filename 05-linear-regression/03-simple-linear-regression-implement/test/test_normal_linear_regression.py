import matplotlib.pyplot as plt
import numpy as np
from package.LinearRegression import LinearRegression
from package.train_test_split import train_test_split
from sklearn import datasets

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

reg = LinearRegression()
reg.fit_normal(X_train, y_train)

print(reg.coef_)  # 截距
# [-1.20354261e-01  3.64423279e-02 -3.61493155e-02  5.12978140e-02
# -1.15775825e+01  3.42740062e+00 -2.32311760e-02 -1.19487594e+00
# 2.60101728e-01 -1.40219119e-02 -8.35430488e-01  7.80472852e-03
# -3.80923751e-01]

print(reg.intercept_)  # 参数
# 34.11739972320798

print(reg.score(X_test, y_test))
# 0.8129794056212908
