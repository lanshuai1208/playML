import matplotlib.pyplot as plt
import numpy as np
from package.train_test_split import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression  # 使用scikit-learn的线性回归类
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.neighbors import KNeighborsRegressor  # 使用scikit-learn的KNN回归类

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

reg = LinearRegression()
reg.fit(X_train, y_train)  # scikit-learn中线性回归模型使用 fit 方法训练，没有多种fit

print(reg.coef_)  # 截距

print(reg.intercept_)  # 参数

print(reg.score(X_test, y_test))  # R2值


knn_reg = KNeighborsRegressor()  # 使用KNN回归器,参数默认是5
knn_reg.fit(
    X_train, y_train
)  # scikit-learn 使用固定格式：1. 创建算法实例 2. 训练fit 3. 预测predict或者查看评价标准score
knn_reg.score(X_test, y_test)  # 0.586 比如线性回归，可以定义超参数来优化


# 使用GridSearchCV搜索最优的超参数组合

param_grid = [
    {"weights": ["uniform"], "n_neighbors": [i for i in range(1, 11)]},
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1, 6)],
    },
]
knn_reg = KNeighborsRegressor()
grid_search = GridSearchCV(  # 使用了交叉验证的方式
    knn_reg, param_grid, n_jobs=-1, verbose=1
)  # knn_reg：回归器；
# param_grid：要搜索的超参数对应的数组
# n_jobs: 是否要并行处理，要的话要用多少核，这里用计算机所有核-1
# verbose：输出详细程度，这里设置为级别1，级别越高，输出内容越详细
grid_search.fit(X_train, y_train)

print(grid_search.best_score_)  # 0.652216494152461
print(grid_search.best_params_)  # {'n_neighbors': 7, 'p': 1, 'weights': 'distance'}
print(grid_search.best_estimator_.score(X_test, y_test))
