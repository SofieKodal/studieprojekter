import numpy as np

def BFGSmethod_line(H, maxit, fundfun, x0, *args):

    # Solver settings and info
    tol = 1.0e-10
    rho = 0.5
    c = 0.1

    # Initial iteration
    x = x0
    it = 0
    f, df, _ = fundfun(x, *args)
    converged = (np.linalg.norm(df, np.inf) <= tol)
    nfun = 1

    # Store data for plotting
    X = [x]
    F = [f]
    dF = [df]
    alpha_vals = []

    I = np.eye(len(x))

    # Main loop of BFGS
    while not converged and (it < maxit):
        it += 1

        p = -np.dot(H, df)

        # backtracking line search
        alpha = 1
        fnew = fundfun(x + alpha*p, *args)[0]
        while fnew > (f + c*alpha*np.dot(df.T, p)):
            alpha *= rho
            fnew = fundfun(x + alpha*p, *args)[0]
        alpha_vals.append(alpha)

        xnew = x + alpha*p
        fnew, dfnew, _ = fundfun(xnew, *args)

        s = xnew - x
        y = dfnew - df
        rhok = 1 / np.dot(y.T, s)
        H = np.dot((I - rhok*np.outer(s, y)), np.dot(H, (I - rhok*np.outer(y, s)))) + rhok*np.outer(s, s)

        x = xnew
        f = fnew
        df = dfnew

        converged = (np.linalg.norm(df, np.inf) <= tol)
        nfun += 1

        # Store data for plotting
        X.append(x)
        F.append(f)
        dF.append(df)

    # Prepare return data
    if not converged:
        x = None 

    # Prepare return status
    stat = {
        'converged': converged,
        'iter': it,
        'nfun': nfun,
        'X': np.array(X),
        'F': np.array(F),
        'dF': np.array(dF),
        'alpha': np.array(alpha_vals)
    }

    return x, stat
