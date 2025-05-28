import numpy as np

def steepestdescent_line_method(fundfun, x0, tol=1.0e-6, rho=0.5, c=0.1, **kwargs):
    # Convert x0 to a numpy array for consistency
    x = np.array(x0, ndmin=1)
    is_scalar = x.size == 1

    # Solver settings and info
    maxit = 2000

    # Initialize
    stat = {"converged": False, "nfun": 0, "iter": 0, "alpha": [], "X": [x.copy()], "F": [], "normdF": []}
    it = 0
    f, df = fundfun(x, **kwargs)
    norm_df = np.linalg.norm(df)
    converged = (norm_df <= tol)
    stat["nfun"] += 1
    
    # Store data for plotting
    stat["F"].append(f)
    stat["normdF"].append(norm_df)
    
    # Main loop of steepest descent
    while not converged and (it < maxit):
        it += 1
        
        # Calculate the descent direction
        p = -df / np.linalg.norm(df, 2)
        
        # Backtracking line search
        alpha = 1
        fnew, _ = fundfun(x + alpha * p, **kwargs)
        while fnew > (f + c * alpha * np.dot(df.T, p)):
            alpha = rho * alpha
            fnew, _ = fundfun(x + alpha * p, **kwargs)
        stat["alpha"].append(alpha)
        
        x = x + alpha * p
        f, df = fundfun(x, **kwargs)
        norm_df = np.linalg.norm(df)
        converged = (norm_df <= tol)
        stat["nfun"] += 1
        
        # Store data for plotting
        stat["X"].append(np.copy(x))
        stat["F"].append(f)
        stat["normdF"].append(norm_df)
    
    stat['iter'] = it
    # Prepare return data
    if not converged:
        stat['converged'] = converged
        return None, stat
    stat['converged'] = converged
    # Convert the solution back to a scalar if the input was a scalar
    x_result = x[0] if is_scalar else x
    return x_result, stat
