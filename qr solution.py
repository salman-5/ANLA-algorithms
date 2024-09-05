import numpy as np

def implicit_qr(A):
    eps = np.spacing(1)
    m, n = A.shape
    W = np.zeros_like(A, dtype=np.complex128)
    W=W.T
    if m>n:
        m_n=m
    else:
        m_n=n
    R = np.zeros_like(A, dtype=np.complex128)
    for k in range(n):
        if(k==0):
            x = A[0:m, 0:n]
        else:
            x = A[1:m, 1:n]
        if(n==k+1):
            if(x[0]>0):
                sign=1
            else: 
                sign=-1
        else:
            if(x[0,0]>0):
                sign=1
            else: 
                sign=-1
        temp=np.eye(m-k)[0]

        v1= x.T[0,:] + sign*np.dot(np.linalg.norm(x.T[0]),temp)
        W[k,k:m]=v1/np.linalg.norm(v1)
        h_matrix=np.eye(m_n-k)-(2/np.dot(v1,v1.conj().T))*np.outer(v1.conj().T,v1)
        A=h_matrix @ x
        R[k:m,k:n] = A
    W=W.T
    return W, R

def form_q(W):
    m, n = W.shape
    if m>n:
        m_n=m
    else:
        m_n=n
    Q = np.zeros_like(W, dtype=np.complex128)
    v1=W.T[0]
    h_matrix=np.eye(m_n)-(2/np.dot(v1,v1.conj()))*np.outer(v1.conj(),v1)
    Q=h_matrix
    for k in range(1,n):
        v1=W[k:m,k].T
        h=np.eye(m_n, dtype=np.complex128)
        h[k:m_n,k:m_n]=np.eye(m_n-k)-(2/np.dot(v1,v1.conj()))*np.outer(v1.conj(),v1)
        Q = Q @ h
    return Q
