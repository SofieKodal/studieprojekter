import matplotlib.pyplot as plt
import numpy as np
from steepestdescent_line import steepestdescent_line_method
from newton_line import newton_line_method
from MvFun import MvFun

#%% Steepest Descent with line search
x0 = np.array([10, 1])
xopt, stat = steepestdescent_line_method(MvFun, x0)

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


#%% Newton with line search
x0 = np.array([10, 1])
xopt, stat = newton_line_method(MvFun, x0)

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


#%% Print out conclusions
print("Since it is a quadratic function, Newton's method only need 2 iterations.")
print("For this example, there are no difference for Newton's method with or without line search.")
print("But we need note that with line search Newton's method can have global convergence.")
