import numpy as np

def steepestdescent_method(alpha, fundfun, x0, *args):
    # Convert x0 to a numpy array for consistency
    x = np.array(x0, ndmin=1)
    is_scalar = x.size == 1

    # Solver settings and info
    maxit = 100 * (1 if is_scalar else len(x))
    tol = 1.0e-10

    # Initial iteration
    stat = {"converged": False, "nfun": 0, "iter": 0, "X": [x.copy()], "F": [], "dF": []}
    it = 0
    f, df, _ = fundfun(x, *args)
    converged = (np.linalg.norm(df, np.inf) <= tol)

    # Store data for plotting
    stat["F"].append(f)
    stat["dF"].append(df.copy())

    # Main loop of steepest descent
    while not converged and (it < maxit):
        it += 1

        p = - df / np.linalg.norm(df)
        x = x + alpha * p

        f, df, _ = fundfun(x, *args)
        converged = (np.linalg.norm(df, np.inf) <= tol)

        # Store data for plotting
        stat['X'].append(x.copy())
        stat['F'].append(f)
        stat['dF'].append(df.copy())
        stat['nfun'] += 1

    stat['iter'] = it
    # Prepare return data
    if not converged:
        stat['converged'] = converged
        return None, stat
    stat['converged'] = converged
    # Convert the solution back to a scalar if the input was a scalar
    x_result = x[0] if is_scalar else x
    return x_result, stat

