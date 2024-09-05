import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import qr

def cg(A, b, tol=1e-12):
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    r_b = [1]
    # todo

    return x, r_b


def arnoldi_n(A, Q, P):
    # n-th step of arnoldi
    m, n = Q.shape
    q = np.zeros(m, dtype=Q.dtype)
    h = np.zeros(n + 1, dtype=A.dtype)
    # todo

    return h, q


def gmres(A, b, P=np.eye(0), tol=1e-12):
    m = A.shape[0]
    if P.shape != A.shape:
        # default preconditioner P = I
        P = np.eye(m)
    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]
    # todo

    return x, r_b


def gmres_givens(A, b, P=np.eye(0), tol=1e-12):
    m = A.shape[0]
    if P.shape != A.shape:
        # default preconditioner P = I
        P = np.eye(m)
    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]
    # todo

    return x, r_b
