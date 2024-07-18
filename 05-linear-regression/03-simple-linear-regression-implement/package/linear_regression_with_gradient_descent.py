# 在线性回归模型中使用梯度下降法

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(666)  # 添加随机种子，让随机具有可重复性
x = 2 * np.random.random(size=100)  # 模拟一维向量，只有一个特征，长度为100
y = x * 3.0 + 4.0 + np.random.normal(size=100)  # 后边是一个噪音，均值为0，方差为1

X = x.reshape(
    -1, 1
)  # 单维模拟多维，参数意思是100行，1列。行设置为-1，就是说具体几行根据列来看，这里列为1，行就是100/1=100

plt.scatter(x, y)
plt.show()


# 使用他梯度下降法你和线性回归
# X_b是第一列增加了1的矩阵
def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
    except Exception as ex:
        return float("inf", ex)


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


X_b = np.hstack([np.ones((len(x), 1)), x.reshape(-1, 1)])
initial_theta = np.zeros(X_b.shape[1])  # X_b.shape[1] X矩阵的列数，也就是待定θ的数量
eta = 0.01

theta = gradient_descent(X_b, y, initial_theta, eta)

print(theta)
