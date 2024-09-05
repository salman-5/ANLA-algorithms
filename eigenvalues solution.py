import numpy as np

def solve(A,b):
    m,n=A.shape
    w=np.zeros(m)
    Q,R=np.linalg.qr(A)
    test=np.dot(Q.T,b)
    for i in range(m-1,-1,-1):
        s= np.dot(R[i][i+1:],w[i+1:])
        w[i]=(test[i]-s)/R[i,i]
    return w

def gershgorin(A):
    λ_min, λ_max = 0,0
    m = A.shape[0]
    for i in range(m):
        center = A[i, i]
        radius = np.sum(np.abs(A[i, :])) - np.abs(center)
        λ_min = min(λ_min, center - radius)
        λ_max = max(λ_max, center + radius)

    return λ_min, λ_max


def power(A, v0):
    v = v0.copy()
    λ = 0
    err = []
    while np.max(np.abs(np.dot(A,v) - λ * v)) >= 1e-13:
        w = np.dot(A, v)
        v= w/ np.linalg.norm(w)
        λ = v.T.dot( ( np.dot(A ,v)))
        error = np.max(np.abs(np.dot(A,v) - λ * v))
        err.append(error)
    return v, λ, err




def inverse(A, v0, μ):
    v = v0.copy()
    λ = 0
    m=A.shape[0]
    err = []
    w=np.zeros(m)
    while True:

        w= solve(A-μ*np.eye(m),v)
        v= w/ np.linalg.norm(w)
        λ = v.T.dot( ( np.dot(A ,v)))
        error = np.max(np.abs(np.dot(A,v) - λ * v))
        err.append(error)
        if error <= 1e-13:
            break
        
    return v, λ, err


def rayleigh(A, v0):
    v = v0.copy()
    λ = 0
    m=A.shape[0]
    λ = v.T.dot( ( np.dot(A ,v)))
    err = []
    while True:

        w=solve(A- λ*np.eye(m),v)
        v= w/ np.linalg.norm(w)
        λ = v.T.dot( ( np.dot(A ,v)))
        error = np.max(np.abs(np.dot(A,v) - λ * v))
        err.append(error)

        if error <= 1e-13:
            break
        
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
    

