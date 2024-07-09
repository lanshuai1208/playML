import matplotlib.pyplot as plt
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.0, 3.0, 2.0, 3.0, 5.0])

# 画出散点图
plt.scatter(x, y)
plt.axis([0, 6, 0, 6])

# 最小二乘法
# 1. 求x,y平均值
x_mean = np.mean(x)
y_mean = np.mean(y)

# 2. 计算a
num = 0.0
d = 0.0

for x_i, y_i in zip(
    x, y
):  # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2

a = num / d

# 3. 计算b
b = y_mean - a * x_mean

# 4. 绘图
y_hat = a * x + b
plt.plot(x, y_hat, color="r")
plt.show()

# 使用最小二乘法得出参数的函数预测
x_predict = 6
y_predict = a * x_predict + b
print(y_predict)
