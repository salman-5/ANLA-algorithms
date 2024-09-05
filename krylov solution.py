import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import qr

def cg(A, b, tol=1e-12):
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    b_norm = np.linalg.norm(b)
    r_b = [1]
    r = b.copy()
    p = r.copy()
    for i in range(m):
        alpha = np.dot(r,r) / np.matmul(p.T, np.matmul(A, p))
        x += alpha * p
        r_prev_inner = np.dot(r,r)
        r -= alpha * (np.matmul(A,p))
        beta = np.dot(r,r) / r_prev_inner
        p = r + (beta * p)
        r_b.append(np.linalg.norm(r)/b_norm)
        if r_b[-1] < tol:
            break

    return x, r_b


def arnoldi_n(A, Q, P):
    # n-th step of arnoldi
    m,n= A.shape
    n_1=Q.shape[1]
    q = np.zeros(m, dtype=Q.dtype)
    h = np.zeros(n + 1, dtype=A.dtype)
    v= solve_triangular(P,np.dot(A,Q[:,n_1-1]))
    for i in range(n_1):
        h[i]= Q[:,i].conj()@v
        v= v - h[i]*Q[:,i]

    h[n_1]=np.linalg.norm(v)
    q=v/h[n_1]
    return h, q


def gmres(A, b, P=np.eye(0), tol=1e-12):
    m,n = A.shape
    if P.shape != A.shape:
        # default preconditioner P = I
        P = np.eye(m)
    x = np.zeros(m, dtype=b.dtype)
    Q= np.zeros((m,1),dtype=A.dtype)
    Hn= np.zeros(n+1,dtype=A.dtype)
    q= np.zeros(m, dtype=A.dtype)
    q= solve_triangular(P,b)
    q=q/np.linalg.norm(q)
    r_b= [np.linalg.norm(b)]
    Q=np.expand_dims(q,axis=1)
    res=np.linalg.norm(b)
    k=0
    while (r_b[k]>tol):
        h,q=arnoldi_n(A,Q,P)
        Q = np.append(Q, np.expand_dims(q, axis=1), axis=1)
        if(len(Hn.shape)==1):
            Hn=np.expand_dims(h, axis=1)
        else:
            Hn = np.append(Hn, np.expand_dims(h, axis=1), axis=1)
        e1=np.zeros(k+1,dtype=A.dtype)
        e1[0]=1
        q_r,R= qr(Hn[:k+2,:k+1])
        q_r=q_r[:k+1,:k+1]
        g=np.dot(q_r.T,res*e1.T)
        g=g[:k+1]
        y=solve_triangular(R,g)
        x=Q[:,:k+1]@y
        r_b.append(np.linalg.norm((Hn[:k+1,:k+1]@y) - (res*e1)))
        k=k+1
    return x, r_b

def givens_qr(H,g):
    R = H
    m,n= H.shape
    for k in range(n):
        for j in range(1,m-k):
            x=R[k:m,k]
            h=np.sqrt(x[0]*x[0]+x[j]*x[j])
            c=x[0]/h 
            s=x[j]/h 
            G=np.eye(m-k,m-k)
            G[0,0]=c
            G[j,j]=c
            G[0,j]=s
            G[j,0]=-s
            R[k:m+1,k:m+1]=np.dot(G,R[k:m+1,k:n+1])
            g.append([c,s])
    set_to_zero = lambda x: np.where(np.abs(x) < 1e-15, 0, x)
    R=set_to_zero(R)
    return g, R
def gmres_givens(A, b, P=np.eye(0), tol=1e-12):
    m,n = A.shape
    if P.shape != A.shape:
        # default preconditioner P = I
        P = np.eye(m)
    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]
    x = np.zeros(m, dtype=b.dtype)
    Q_f= np.zeros(n+1,dtype=A.dtype)
    R_f= np.zeros(1,dtype=A.dtype)


    Q= np.zeros((m,1),dtype=A.dtype)
    Hn= np.zeros(n+1,dtype=A.dtype)
    
    
    q= np.zeros(m, dtype=A.dtype)
    
    q= solve_triangular(P,b)
    r_b= [np.linalg.norm(q)]
    res=np.linalg.norm(q)
    q=q/np.linalg.norm(q)
    Q=np.expand_dims(q,axis=1)


    k=0
    g1=[]
    while (r_b[k]>tol):
        h,q=arnoldi_n(A,Q,P)
        Q = np.append(Q, np.expand_dims(q, axis=1), axis=1)
        if(len(Hn.shape)==1):
            Hn=np.expand_dims(h, axis=1)
        else:
            Hn = np.append(Hn, np.expand_dims(h, axis=1), axis=1)
        temp=Hn[:k+2,:k+1].copy()
        e2=np.zeros(temp.shape[0],dtype=A.dtype)
        e2[0]=1

        temp=np.append(temp,np.expand_dims(res*e2,axis=1),axis=1)
        e1=np.zeros(k+1,dtype=A.dtype)
        e1[0]=1
        q_r,R= givens_qr(temp,g1)
        s=np.array(R[:k+1,:k+1])
        g=np.array([R[:k+1,-1]])
        y=solve_triangular(s,g.T)
        x=Q[:,:k+1]@y
        r_b.append(np.linalg.norm((Hn[:k+1,:k+1]@y).T - (res*e1)))
        k=k+1
    return x, r_b
