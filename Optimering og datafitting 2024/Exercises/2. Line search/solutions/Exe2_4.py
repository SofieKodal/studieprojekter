import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from steepestdescent import steepestdescent_method
from newton import newton_method
from MvFun import MvFun

#%% Steepest Descent
x0 = np.array([10, 1])
alpha = 0.05
xopt, stat = steepestdescent_method(alpha, MvFun, x0)

err = np.sqrt(np.array(stat['X'])[:, 0]**2 + np.array(stat['X'])[:, 1]**2)
norm_df = np.sqrt(np.array(stat['dF'])[:, 0]**2 + np.array(stat['dF'])[:, 1]**2)

plt.figure()
plt.subplot(1, 3, 1)
plt.plot(err)
plt.title('||x_k-x^*||')

plt.subplot(1, 3, 2)
plt.plot(norm_df)
plt.title('||f\'(x_k)||')

plt.subplot(1, 3, 3)
plt.semilogy(stat['F'])
plt.title('f(x_k)')

plt.show()


#%% Newton
x0 = np.array([10, 1])
alpha = 1
xopt, stat = newton_method(alpha, MvFun, x0)

err = np.sqrt(np.array(stat['X'])[:, 0]**2 + np.array(stat['X'])[:, 1]**2)
norm_df = np.sqrt(np.array(stat['dF'])[:, 0]**2 + np.array(stat['dF'])[:, 1]**2)

plt.figure()
plt.subplot(1, 3, 1)
plt.plot(err)
plt.title('||x_k-x^*||')

plt.subplot(1, 3, 2)
plt.plot(norm_df)
plt.title('||f\'(x_k)||')

plt.subplot(1, 3, 3)
plt.semilogy(stat['F'])
plt.title('f(x_k)')

plt.show()
