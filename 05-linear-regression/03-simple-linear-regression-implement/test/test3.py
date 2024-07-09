import time

import numpy as np
from package.SimpleLinearRegression1 import SimpleLinearRegression1
from package.SimpleLinearRegression2 import SimpleLinearRegression2


def timer(func):
    def wrapper(*args, **kwargs):
        # start the timer
        start_time = time.time()
        # call the decorated function
        result = func(*args, **kwargs)
        # remeasure the time
        end_time = time.time()
        # compute the elapsed time and print it
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        # return the result of the decorated function execution
        return result

    # return reference to the wrapper function
    return wrapper


m = 1000000
big_x = np.random.random(size=m)
big_y = big_x * 2.0 + 3.0 + np.random.normal(size=m)

reg1 = SimpleLinearRegression1()  # Execution time: 0.7320215702056885 seconds
reg2 = SimpleLinearRegression2()  # Execution time: 0.009568929672241211 seconds


@timer
def fit1():
    reg1.fit(big_x, big_y)


@timer
def fit2():
    reg2.fit(big_x, big_y)


fit1()
fit2()
