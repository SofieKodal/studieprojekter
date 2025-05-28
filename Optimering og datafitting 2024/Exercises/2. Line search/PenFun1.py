import numpy as np

def PenFun1(x, mu):
    if x > 0:
        f = x - mu*np.log(x)
        df = 1.0 - mu/x
        d2f = mu/(x*x)
        return f, df, d2f
    else:
        raise ValueError("x needs to be positive")
