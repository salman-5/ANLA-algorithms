import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as pl


def tridiag(A):
    m, _ = A.shape  
    for k in range(0, m-2):
        x = A[0:m, k] 
        y=x.copy()
        y[k+2:m]=0
        y[k+1]=np.linalg.norm(A[k+1:m,k])
        w=x-y
        w = w / np.linalg.norm(w)  
        p=np.eye(m)-2*np.outer(w,w.conj())
        A=p@A @p
    set_to_zero = lambda x: np.where(np.abs(x) < 1e-15, 0, x)
    A=set_to_zero(A)
    return A


def QR_alg(T):
    t = []
    m,_ =T.shape
    while(np.abs (T[m-1,m-2])> 1e-12):
        Q,R=np.linalg.qr(T)
        T=R@Q
        t.append(np.abs(T[m-1,m-2]))
    set_to_zero = lambda x: np.where(np.abs(x) < 1e-15, 0, x)
    T=set_to_zero(T)
    return (T, t)

def wilkinson_shift(T):
    μ = 0
    m,_ = T.shape
    
    delta=(T[m-2,m-2]-T[m-1,m-1])*0.5
    if(delta>=0):
        s=1
    else:
        s=-1
    b_m=T[m-1,m-2]
    μ = (T[m-1,m-1]-(s*b_m*b_m))/(np.abs(delta)+ np.sqrt(delta*delta+b_m*b_m))
    return μ


def QR_alg_shifted(T):
    t = []
    m,_ =T.shape
    while(np.abs(T[m-1,m-2])>1e-12):  
        μ = wilkinson_shift(T)
        Q,R=np.linalg.qr(T- np.eye(m)*μ)
        T=(R @Q )+  np.eye(m)*μ
        t.append(np.abs(T[m-1,m-2]))

    set_to_zero = lambda x: np.where(np.abs(x) < 1e-15, 0, x)
    T=set_to_zero(T)
    return (T, t)



def QR_alg_driver(A, shift):
    all_t = []
    Λ = []
    m,_=A.shape
    T=tridiag(A)
    a=T.copy()
    if(shift==True):
        for i in range(m,1,-1):
            a,b=QR_alg_shifted(a[0:i,0:i])
            all_t=all_t+b
            Λ.append(a[-1,-1])
        Λ.append(a[0,0])        
    else:
        for i in range(m,1,-1):
            a,b=QR_alg(a[0:i,0:i])
            all_t=all_t+b
            Λ.append(a[-1,-1])
        Λ.append(a[0,0])


    Λ = np.sort(Λ)
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