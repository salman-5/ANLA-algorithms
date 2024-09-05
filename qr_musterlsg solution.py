import numpy as np

def givens_qr(H):
    G = []
    R = H
    m,n= H.shape
    if m>n:
        max_size=m
        min_size=n
    else:
        max_size=n
        min_size=m
    
    i=1
    
    for j in range(0,R.shape[1]):
            [c,s] = givens_cs(R[i-1][j],R[i][j]) 
            rot=np.array([[c, -s],[s, c]])
            Temp=np.eye(max_size)
            Temp[i-1:i+1,i-1:i+1]= rot
            R=Temp.conj().T@ R
            G.append([c,s])
            i+=1
    G=np.array(G)
    return (G, R)
def givens_cs(a,b):
    if np.abs(a) <= np.abs(b):
        if(b==0):
            t=a
        else:
            t=a/b
        s=np.sign(b)/(np.sqrt(1+t*t))
        c=s*t
    else:
        t=b/a
        c=np.sign(a)/(np.sqrt(1+t*t))
        s=c*t
    
    return [c,s]

def form_q(G):

    max_size=len(G)
    Q = np.eye(max_size+1 )
    i=0
    for c,s in G:
        Temp=np.eye(max_size+1)
        Temp[i][i]=c
        Temp[i][i+1]=-s
        Temp[i+1][i]=s
        Temp[i+1][i+1]=c
        print(Temp)
        i+=1
        Q=Q@ Temp
    print(Q)
    return Q
