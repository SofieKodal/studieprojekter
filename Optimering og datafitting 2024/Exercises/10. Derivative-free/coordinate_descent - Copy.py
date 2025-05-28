import numpy as np

def coordinate_descent(fundfun, gamma0, x0, maxit=None, *args):
    # Coordinate search method
    '''
    Inputs: 
    fundfun:    a function handle, which evaluates the objective function.
    gamma0:     the step length.
    x0:         the starting point.
    maxit:      the maximum number of iterations.
    args:       the parameters in fundfun.

    Outputs:
    x_result:   the solution, if the algorithm converges.
    stat:       a structure array. 
                stat.converged shows if the algorithm converges.
                stat.iter gives the number of iterations.
                stat.X saves all iterate including the starting point.
                stat.F saves all objective function values.
                stat.nfun saves the number of function evaluations.
    '''

    # Determine if x0 is a scalar or an array
    x = np.array(x0, ndmin=1)
    is_scalar = x.size == 1

    # Solver settings and info
    maxit = maxit if maxit is not None else 10000
    tol = 1.0e-6
    n = len(x0)
    I = np.eye(n)

    stat = {"converged": False, "nfun": 0, "iter": 0, "X": [x.copy()], "F": []}

    # Initial iteration
    gamma = gamma0
    it = 0
    f = fundfun(x, *args)
    converged = (gamma <= tol)
    stat["nfun"] += 1
    stat['F'].append(f)

    # Main loop of coordinate search
    while not converged and it < maxit:
        it += 1

        # TODO -- evaluate the function values for all points in D_k,
        #         then update x or reduce gamma
        # ======================================


        # ======================================

        converged = (gamma <= tol)

        stat['X'].append(x)
        stat['F'].append(f)

    stat['iter'] = it
    # Prepare return data
    if not converged:
        stat['converged'] = converged
        return None, stat
    stat['converged'] = converged
    # Convert the solution back to a scalar if the input was a scalar
    x_result = x[0] if is_scalar else x
    return x_result, stat
