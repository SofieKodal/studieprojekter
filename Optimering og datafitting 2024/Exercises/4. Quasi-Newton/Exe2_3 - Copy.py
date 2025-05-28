import numpy as np
import pandas as pd

from scipy.optimize import minimize
from Myfun import Myfun


x0 = np.array([1,4])

all_x_i = [x0[0]]
all_y_i = [x0[1]]
all_f_i = [Myfun(x0)[0]]
def store(X):
    x, y = X        
    all_x_i.append(x)
    all_y_i.append(y)
    all_f_i.append(Myfun(X)[0])
    return all_x_i, all_y_i, all_f_i

minimize(Myfun, x0, method="BFGS", jac=True, callback=store, options={'maxiter': 8})

table = list(zip(all_x_i, all_y_i, all_f_i))

# Create a DataFrame
df = pd.DataFrame(table, columns=['x1_k', 'x2_k', 'f(x_k)'])

# Display the DataFrame
print(df)