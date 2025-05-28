import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from BFGSmethod_line import BFGSmethod_line_method
from Myfun import Myfun

x0 = np.array([1,4])
H = np.eye(len(x0))
maxit = 8
xopt, stat = BFGSmethod_line_method(H, Myfun, x0, maxit)

k = np.arange(stat['iter']+1)
err = np.abs(np.array(stat['X'])-1)

table = list(zip(k, stat['X'], stat['F']))

# Create a DataFrame
df = pd.DataFrame(table, columns=['iter', 'x_k', 'f(x_k)'])

# Display the DataFrame
print(df)

# Plot contour
x1 = np.arange(-4, 6.05, 0.05)
x2 = np.arange(-1, 9.05, 0.05)
X, Y = np.meshgrid(x1, x2)
F = X**4 - 2*X**2*Y + X**2 + Y**2 - 2*X + 5

fig, ax = plt.subplots()
v = np.arange(3, 20.5, 0.5)
c = ax.contour(X, Y, F, v, linewidths=2)
norm = matplotlib.colors.Normalize(vmin=c.cvalues.min(), vmax=c.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = c.cmap)
sm.set_array([])
# fig.colorbar(sm, ticks=c.levels)
fig.colorbar(sm, ax=ax)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.gca().set_aspect('equal', adjustable='box')

# Plot iterations
plt.plot(np.array(stat['X'])[:,0], np.array(stat['X'])[:,1], 'r*')
plt.show()

