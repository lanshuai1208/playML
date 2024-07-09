import matplotlib.pyplot as plt
import numpy as np
from package.SimpleLinearRegression1 import SimpleLinearRegression1

reg1 = SimpleLinearRegression1()

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.0, 3.0, 2.0, 3.0, 5.0])

x_predict = np.array(
    [
        6.0,
        7.0,
        8.0,
        9.0,
    ]
)

reg1.fit(x, y)

y_predict = reg1.predict(x_predict)
print(y_predict)

xs = np.append(x, x_predict)
ys = np.append(y, y_predict)

plt.scatter(xs, ys)
plt.axis([0, 10, 0, 10])
plt.plot(xs, reg1.predict(np.append(x, x_predict)), color="r")
plt.show()
