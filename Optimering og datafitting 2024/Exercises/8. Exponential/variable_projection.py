import numpy as np
from linearLSQ import linearLSQ

def variable_projection_method(fun_All, a0, maxit=None, *args):
    # Determine if x0 is a scalar or an array
    a = np.array(a0, ndmin=1)
    is_scalar = a.size == 1

    # Solver settings and info
    maxit = maxit if maxit is not None else 100 * (1 if is_scalar else len(a0))
    tol = 1.0e-10

    stat = {"converged": False, "nfun": 0, "iter": 0, "X": [], "F": [], "dF": []}

    # Initial iteration
    it = 0

    ra, Ja, ca = fun_All(a, *args)
    f = 0.5 * np.linalg.norm(ra, 2)**2
    df = np.dot(Ja.T, ra).flatten()

    converged = np.linalg.norm(df, np.inf) <= tol

    # Initial lambda
    lambda_val = np.linalg.norm(Ja.T @ Ja, 2)

    # Store data for plotting
    stat['X'].append(np.concatenate([ca, a]))
    stat['F'].append(f)
    stat['dF'].append(df)

    # Main loop of variable projection method
    while not converged and it < maxit:
        it += 1

        # Calculate the search direction by solving a linear LSQ problem    
        A = np.vstack([Ja, np.sqrt(lambda_val) * np.eye(len(a))])
        b = np.hstack([-ra, np.zeros_like(a)])
        p = linearLSQ(A, b)

        # Update the iterate, Jacobian, residual, f, df
        a_new = a + p

        # Update the Lagrange parameter lambda
        ra_new, Ja_new, ca_new = fun_All(a_new, *args)
        f_new = 0.5 * np.linalg.norm(ra_new, 2)**2

        rho = (f - f_new) / (0.5 * (p @ (lambda_val * p - Ja.T @ ra)))
        if rho > 0.75:
            lambda_val /= 3
        elif rho < 0.25:
            lambda_val *= 2

        # Accept or reject a_new
        if rho > 0:
            a = a_new
            ra = ra_new
            f = f_new
            Ja = Ja_new
            ca = ca_new
            df = np.dot(Ja.T, ra).flatten()

        converged = np.linalg.norm(df, np.inf) <= tol
        stat['nfun'] += 1

        # Store data for plotting
        x = np.concatenate([ca, a])
        stat['X'].append(x)
        stat['F'].append(f)
        stat['dF'].append(df)

    stat['iter'] = it
    # Prepare return data
    if not converged:
        stat['converged'] = converged
        return None, stat
    stat['converged'] = converged
    # Convert the solution back to a scalar if the input was a scalar
    x_result = x[0] if is_scalar else x
    return x_result, stat