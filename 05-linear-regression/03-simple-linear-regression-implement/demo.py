import matplotlib.pyplot as plt
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.0, 3.0, 2.0, 3.0, 5.0])

plt.scatter(x, y)


plt.axis([0, 6, 0, 6])
