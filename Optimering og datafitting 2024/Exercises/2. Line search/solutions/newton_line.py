import numpy as np

def newton_line_method(fundfun, x0, tol=1.0e-10, rho=0.5, c=0.1, *args):
    # Convert x0 to an array for consistency
    x = np.array(x0, ndmin=1)
    is_scalar = x.size == 1

    # Solver settings and info
    maxit = 100 * (1 if is_scalar else len(x))

    # Initialize
    stat = {"converged": False, "nfun": 0, "iter": 0, "alpha": [], "X": [x.copy()], "F": [], "dF": []}
    it = 0
    f, df, d2f = fundfun(x, *args)
    converged = (np.linalg.norm(df, np.inf) <= tol)
    stat["nfun"] += 1
    
    # Store data for plotting
    stat["F"].append(f)
    stat["dF"].append(df.copy())
    
    # Main loop of Newton method
    while not converged and (it < maxit):
        it += 1
        
        # Compute the descent direction
        if is_scalar:
            p = -df / d2f
        else:
            p = -np.linalg.solve(d2f, df)
        
        # backtracking line search
        alpha = 1
        fnew, _, _ = fundfun(x + alpha * p, *args)
        while fnew > (f + c * alpha * np.dot(df.T, p)) and alpha>0.01:
            alpha = rho * alpha
            fnew, _, _ = fundfun(x + alpha * p, *args)
        stat["alpha"].append(alpha)
        
        x = x + alpha * p
        f, df, d2f = fundfun(x, *args)
        converged = (np.linalg.norm(df, np.inf) <= tol)
        stat["nfun"] += 1
        
        # Store data for plotting
        stat["X"].append(np.copy(x))
        stat["F"].append(f)
        stat["dF"].append(df.copy())
    
    stat['iter'] = it
    # Prepare return data
    if not converged:
        stat['converged'] = converged
        return None, stat
    stat['converged'] = converged
    # Convert the solution back to a scalar if the input was a scalar
    x_result = x[0] if is_scalar else x
    return x_result, stat

