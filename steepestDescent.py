
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def steepestDescent(par, A, d, c, verbose=False, alpha=0.01, tol=1e-6):

    # par = np.array([4, 5])
    # A = np.array([[10, -6], [-6, 10]])
    # d = np.array([4, 4])
    # c = 0
    # verbose = True
    # alpha = 0.01
    # tol = 1e-6

    l, P = np.linalg.eig(A)
    if any(l <= 0):
        sys.exit("Not all eigenvalues greater than zero.\n")

    par_k = par 
    k = 1
    ans = par_k 

    while True: 

        # Calculate gradient of quadratic function directly. 
        g = np.dot(A, par_k) + d
        par_k1 = par_k - alpha * g
        epsilon = np.linalg.norm(par_k1 - par_k, ord=2)
        if epsilon < tol:
            break
        else: 
            par_k = par_k1
            k = k + 1
        ans = np.vstack((ans, par_k))

        if verbose == True: 
            print(k, ": (", round(par_k1[0], 4), ", ", round(par_k1[1], 4), ")")

    # Clean up results for output. 
    ans = pd.DataFrame(ans, columns = ['x', 'y'])
    ans['iter'] = range(1, ans2.shape[0] + 1)

    # Round the answer. 
    par_k1 = np.round(par_k1, 4)
    return par_k1, ans

par = np.array([4, 5])
A = np.array([[10, -6], [-6, 10]])
d = np.array([4, 4])
c = 0
xStar, conv = steepestDescent(par, A, d, c)