import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Given data
t = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])
y = np.array([0.80, 1.23, 1.15, 1.48, 2.17])

# Create grid for contour plotting
x1 = np.linspace(0, 0.5, 51)
x2 = np.linspace(1, 1.5, 51)
X, Y = np.meshgrid(x1, x2)

# Calculate F2, F1, Finf
F2 = np.zeros_like(X)
F1 = np.zeros_like(X)
tmp = np.zeros((X.shape[0], X.shape[1], len(t)))

for i in range(len(t)):
    F2 += (t[i]*X + Y - y[i])**2
    F1 += np.abs(t[i]*X + Y - y[i])
    tmp[:,:,i] = np.abs(t[i]*X + Y - y[i])

F2 = np.sqrt(F2)
Finf = np.max(tmp, axis=2)

# Contour levels
k = np.arange(10)
v1 = 0.78 + 0.15*k
v2 = 0.4 + 0.1*k
vinf = 0.22 + 0.1*k

# Plotting

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
ax = axs[0]  # Access the first subplot Axes object
c = ax.contour(X, Y, F2, v2, linewidths=2)
norm = matplotlib.colors.Normalize(vmin=c.cvalues.min(), vmax=c.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = c.cmap)
# fig.colorbar(sm, ticks=c.levels)
fig.colorbar(sm, ax=ax)
ax.set_aspect('equal', 'box')
ax.axis('image')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_title('l_2')

ax = axs[1]  # Access the first subplot Axes object
c = ax.contour(X, Y, F1, v1, linewidths=2)
norm = matplotlib.colors.Normalize(vmin=c.cvalues.min(), vmax=c.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = c.cmap)
# fig.colorbar(sm, ticks=c.levels)
fig.colorbar(sm, ax=ax)
ax.set_aspect('equal', 'box')
ax.axis('image')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_title('l_1')

ax = axs[2]  # Access the first subplot Axes object
c = ax.contour(X, Y, Finf, vinf, linewidths=2)
norm = matplotlib.colors.Normalize(vmin=c.cvalues.min(), vmax=c.cvalues.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap = c.cmap)
# fig.colorbar(sm, ticks=c.levels)
fig.colorbar(sm, ax=ax)
ax.set_aspect('equal', 'box')
ax.axis('image')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_title('l_∞')

plt.tight_layout()
plt.show()
