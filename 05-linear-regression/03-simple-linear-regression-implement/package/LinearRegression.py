import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


class LinearRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None  # 系数 coefficients
        self.intercept_ = None  # 截距 intercept
        self._theta = None  # θ向量

    def fit_normal(self, X_train, y_train):  # 使用多元线性回归的正规方程解来训练模型
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack(
            [np.ones((len(X_train), 1)), X_train]
        )  # hstack 横向上多加一列；np.ones 获得一个全1的n维数组

        self._theta = (
            np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        )  # 使用正规方程解来获取θ向量；
        # 矩阵.T：求矩阵的转置矩阵；
        # np.linalg.inv：求逆矩阵；
        # 矩阵.dot(矩阵)：矩阵点乘

        self.intercept_ = self._theta[0]  # 截距就是θ向量第一个元素
        self.coef_ = self._theta[1:]  # 系数就是后边的元素

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except Exception as ex:
                return ex

        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert (
            self.intercept_ is not None and self.coef_ is not None
        ), "must fit before predict!"
        assert X_predict.shape[1] == len(
            self.coef_
        ), "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"


lin_reg = LinearRegression()

np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3.0 + 4.0 + np.random.random(size=100)


X = x.reshape(-1, 1)

lin_reg.fit_gd(X, y)

print(lin_reg.intercept_, lin_reg.coef_)


def y_hat(x):
    return x * lin_reg.coef_[0] + lin_reg.intercept_


plt.scatter(x, y)
plt.plot(x, y_hat(x), color="r")
plt.show()
