import numpy as np

def MvFun(x):
    f = 0.5*x[0]**2 + 5*x[1]**2
    df = np.array([x[0], 10*x[1]])
    d2f = np.array([[1, 0], [0, 10]])
    return f, df, d2f
