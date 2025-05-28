import matplotlib.pyplot as plt
import pandas as pd
from steepestdescent import steepestdescent_method
from newton import newton_method
from PenFun1 import PenFun1

#%% Steepest Descent
x0 = 0.01    # 0.5, 5
mu = 1.0
alpha = 0.1   # 1, 0.01
xopt, stat = steepestdescent_method(alpha, PenFun1, x0, mu)

k = list(range(stat['iter']+1))
err = [abs(x-1) for x in stat['X']]

table = list(zip(k, stat['X'], err, [abs(df) for df in stat['dF']], stat['F']))

# Create a DataFrame
df = pd.DataFrame(table, columns=['iter', 'x_k', 'err', '|f\'(x_k)|', 'f(x_k)'])

# Display the DataFrame
print(df)

plt.figure()
plt.subplot(1, 3, 1)
plt.semilogy(err)
plt.title('|x_k-x^*|')

plt.subplot(1, 3, 2)
plt.plot([abs(df) for df in stat['dF']])
plt.title('f\'(x_k)')

plt.subplot(1, 3, 3)
plt.plot(stat['F'])
plt.title('f(x_k)')

plt.show()


#%% Newton
x0 = 0.1
mu = 1.0
alpha = 1
xopt, stat = newton_method(alpha, PenFun1, x0, mu)

k = list(range(stat['iter']+1))
err = [abs(x-1) for x in stat['X']]

table = list(zip(k, stat['X'], err, [abs(df) for df in stat['dF']], stat['F']))

# Create a DataFrame
df = pd.DataFrame(table, columns=['iter', 'x_k', 'err', '|f\'(x_k)|', 'f(x_k)'])

# Display the DataFrame
print(df)

plt.figure()
plt.subplot(1, 3, 1)
plt.semilogy(err)
plt.title('|x_k-x^*|')

plt.subplot(1, 3, 2)
plt.plot(stat['dF'])
plt.title('f\'(x_k)')

plt.subplot(1, 3, 3)
plt.plot(stat['F'])
plt.title('f(x_k)')

plt.show()
