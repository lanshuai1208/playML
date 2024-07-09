import numpy as np


class SimpleLinearRegression1:
    def __init__(self):
        # 初始化 Simple Linear Regression 模型
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        # 根据训练数据集x_train 和 y_train 训练SimpleLinearRegression模型

        # 进行一些断言保证用户传进来的数据是合法的
        assert (
            x_train.ndim == 1
        ), "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(
            y_train
        ), "the size of x_train must be equal to the size of y_train"

        # 最小二乘法
        # 1. 求x,y平均值
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 2. 计算a
        num = 0.0
        d = 0.0
        for x_i, y_i in zip(
            x_train, y_train
        ):  # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2
        self.a_ = num / d

        # 3. 计算b
        self.b_ = y_mean - self.a_ * x_mean

        return self  # fit函数的规范

    def predict(self, x_predict):  # x_predict 是一系列代预测的x值
        # 给定待预测数据集x_predict，返回表示x_predict的结果向量
        assert (
            x_predict.ndim == 1
        ), "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        # 给定单个待预测数据x，返回x的预测结果值
        return self.a_ * x_single + self.b_

    def __repr__(self):
        # Python 有一个内置的函数叫 repr，它能把一个对象用字符串的形式表达出来以便辨认，这就是“字符串表示形式”。
        # repr 就是通过这个特殊方法来得到一个对象的字符串表示形式的。
        # 如果没有实现，当我们在控制台里打印一个向量的实例时，得到的字符串可能会是 <Vector object at 0x10e100070>
        return "SimpleLinearRegression1"
