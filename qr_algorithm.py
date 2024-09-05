import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as pl


def tridiag(A):
    # todo
    return A


def QR_alg(T):
    t = []
    # todo
    return (T, t)


def wilkinson_shift(T):
    μ = 0
    # todo
    return μ


def QR_alg_shifted(T):
    t = []
    # todo
    return (T, t)


def QR_alg_driver(A, shift):
    all_t = []
    Λ = []
    # todo
    return (Λ, all_t)


if __name__ == "__main__":

    matrices = {
        "hilbert": hilbert(4),
        "diag(1,2,3,4)+ones": np.diag([1, 2, 3, 4]) + np.ones((4, 4)),
        "diag(5,6,7,8)+ones": np.diag([5, 6, 7, 8]) + np.ones((4, 4)),
    }

    fig, ax = pl.subplots(len(matrices.keys()), 2, figsize=(10, 10))

    for i, (mat, A) in enumerate(matrices.items()):
        print(f"A = {mat}")
        Λ,_ = np.linalg.eig(A)
        print(f"Λ = {np.sort(Λ)}\n")
        for j, shift in enumerate([True, False]):
            Λ, conv = QR_alg_driver(A.copy(), shift)
            ax[i, j].semilogy(range(len(conv)), conv, ".-")
            ax[i, j].set_title(f"A = {mat}, shift = {shift}")

    pl.show()