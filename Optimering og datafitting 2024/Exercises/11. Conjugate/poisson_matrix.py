import numpy as np

def poisson_matrix(n):
    # Main diagonal
    main_diag = 4 * np.ones(n**2)
    # Side diagonals
    side_diag = -1 * np.ones(n**2 - 1)
    # Diagonals for the blocks above and below the main diagonal
    block_diag = -1 * np.ones(n**2 - n)
    
    # Create the matrix
    A = np.diag(main_diag) + np.diag(side_diag, -1) + np.diag(side_diag, 1) \
        + np.diag(block_diag, -n) + np.diag(block_diag, n)
    
    # Adjust for the boundary conditions in the side diagonals
    for i in range(n - 1, n**2 - 1, n):
        A[i, i + 1] = 0
        A[i + 1, i] = 0
    
    return A