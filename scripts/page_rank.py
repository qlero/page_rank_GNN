"""
PageRank algorithm with explicit number of iterations.
"""

###########
# IMPORTS #
###########

import numpy as np

#############
# FUNCTIONS #
#############

def page_rank(
    M: np.ndarray, 
    n_iterations: int = 100, 
    d: float = 0.85,
    e: float = 1e-5,
    print_convergence: bool = False
) -> np.ndarray:
    """
    Implements the Google PageRank algorithm using an iterative
    method.

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from
        'j' to 'i', such that for all 'j' sum(i, M_i,j) = 1
    num_iterations : int, optional
        number of iterations, default=100
    d : float, optional
        damping factor, default=0.85
    e : float, optional
        convergence check, default = 1e-5
        
    Returns
    -------
    numpy array
        A vector of ranks such that v_i is the i-th rank 
        from [0, 1], and v sums to 1
    """
    n_nodes = M.shape[1]
    # Initializes a random probability distribution for
    # each page rank (random draw of a matrix + normalization)
    PR = np.random.rand(n_nodes, 1)
    PR = PR / np.linalg.norm(PR, 1)
    # Initializes the weighted adjacency matrix 
    # M_ij = 1/L(p_j) if node j links to node i
    #      = 0 otherwise
    M_hat = (d * M + (1 - d) / n_nodes)
    # Iterates until convergence
    for i in range(n_iterations):
        old_PR = PR.copy()
        PR = M_hat @ PR
        # Checks if converges
        if np.linalg.norm(np.abs(old_PR-PR)) < e:
            if print_convergence:
                print(f"PageRank converged at iteration {i}",
                      f"with epsilon={e}.")
            return PR
    if print_convergence:
        print(f"PageRank did not converge before {n_iterations} its.")
    return PR
