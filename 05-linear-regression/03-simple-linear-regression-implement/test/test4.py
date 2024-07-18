import numpy as np
from package.LinearRegression import LinearRegression

lin_reg = LinearRegression()

x = np.random.random(size=100)
y = x * 2.0 + 3.0 + np.random.random(size=100)

X = x.reshape(-1, 1)

lin_reg.fit_gd(X, y)

print(lin_reg.intercept_, lin_reg.coef_)
