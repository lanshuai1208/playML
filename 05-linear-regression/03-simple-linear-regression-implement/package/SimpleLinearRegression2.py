import numpy as np


class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert (
            x_train.ndim == 1
        ), "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(
            y_train
        ), "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # num = 0.0
        # d = 0.0
        # for x_i, y_i in zip(x_train, y_train):
        #     num += (x_i - x_mean) * (y_i - y_mean)
        #     d += (x_i - x_mean) ** 2

        # 更换为向量点乘
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d

        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        assert (
            x_predict.ndim == 1
        ), "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):

        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1"
