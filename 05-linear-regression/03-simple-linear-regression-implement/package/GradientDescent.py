# 梯度下降法模拟
import matplotlib.pyplot as plt
import numpy as np

plt_x = np.linspace(-1, 6, 141)
print(plt_x)

plt_y = (plt_x - 2.5) ** 2 - 1
plt.plot(plt_x, plt_y)
plt.show()


def J(theta):
    try:
        return (theta - 2.5) ** 2 - 1.0
    except Exception as ex:
        return float("inf", ex)


def dJ(theta):
    return 2 * (theta - 2.5)


eta = 0.1
theta = 0.0
epsilon = 1e-8
theta_history = []
n_iters = 1e4
i_iters = 0


while i_iters < n_iters:
    gradient = dJ(theta)
    lastest_theta = theta
    theta = theta - eta * gradient
    theta_history.append(theta)
    if abs(J(theta) - J(lastest_theta)) < epsilon:
        break

    i_iters += 1

plt.plot(plt_x, plt_y)
plt.plot(np.array(theta_history), J(np.array(theta_history)), color="r", marker="+")
plt.show()
print(theta_history)
print(theta)
print(J(theta))
