import numpy as np


def gershgorin(A):
    λ_min, λ_max = 0,0

    # todo

    return λ_min, λ_max


def power(A, v0):
    v = v0.copy()
    λ = 0
    err = []

    # todo

    return v, λ, err


def inverse(A, v0, μ):
    v = v0.copy()
    λ = 0
    err = []

    # todo

    return v, λ, err


def rayleigh(A, v0):
    v = v0.copy()
    λ = 0
    err = []

    # todo

    return v, λ, err


def randomInput(m):
    #! DO NOT CHANGE THIS FUNCTION !#
    A = np.random.rand(m, m) - 0.5
    A += A.T  # make matrix symmetric
    v0 = np.random.rand(m) - 0.5
    v0 = v0 / np.linalg.norm(v0) # normalize vector
    return A, v0


if __name__ == '__main__':
    pass
    # todo
