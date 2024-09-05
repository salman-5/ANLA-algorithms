# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 12:11:49 2024

@author: arjun
Step by step, we will now put together an Python program that finds all the 
eigenvalues of a real symmetric matrix.
"""
import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as plt


def sign_func(x1):
    """Sign function: Returns +1 if input is greater than zero, -1 otherwise"""
    return (1 if x1>0 else -1)
def tridiag(A):
    """Returns a tridiagonal matrix when a real symmetric matrix is inputted"""
    m,n = np.shape(A)
    if n!= m:
        return ("Input matrix is not symmetric. Try again") 
    R = A.copy()
    for k in range(0,m-2):
        xk = np.copy(R[k+1:,k]) #Take kth column, exlcuding kth rows
        xk_norm = np.linalg.norm(xk,ord=2) #Find norm of the column
        sign = sign_func(xk[0]) 
        ek = np.zeros((m-k-1,1)) 
        ek[0] = 1 #set unit vector
        vk = ek*sign*xk_norm + xk.reshape(-1,1) 
        vk = vk/(np.linalg.norm(vk,ord=2)) #Find vk
        R_temp1 = np.dot(vk.T,R[k+1:,k:])
        R[k+1:,k:] = R[k+1:,k:] - 2*np.dot(vk,R_temp1)
        R_temp2 = np.dot(R[:,k+1:],vk)
        R[:,k+1:] = R[:,k+1:] - 2*np.dot(R_temp2,vk.T) #generate new sub matrix
        R = (R + R.T)/2 #forcing R to be symmetric
        low_value_flags = np.absolute(R) < 1e-12 #rounding off insiginificant values to zero
        R[low_value_flags] = 0
    return R

def wilkinson_shift(A):
    """Function to generate the Wilkinson shift given a square matrix"""
    m,_ = np.shape(A)
    B = np.copy(A[m-2:,m-2:])
    delta = (B[0,0] - B[1,1])*0.5
    sign = sign_func(delta)
    denom = np.absolute(delta) + np.sqrt(delta**2+B[0,1]**2)
    mu = B[1,1] - (sign*B[0,1]**2)/denom
    return mu


def QR_alg(T):
    """Function to implement pure QR algorithm to find eigen values"""
    m,_ = np.shape(T)
    t = [] 
    t.append(np.absolute(T[m-1,m-2]))
    while np.absolute(T[m-1,m-2]) >= 1e-12: #Condition for convergence
        Q, R = np.linalg.qr(T)
        T = np.dot(R,Q)
        t.append(np.absolute(T[m-1,m-2]))
        low_value_flags = np.absolute(T) < 1e-12 #rounding off insiginificant values to zero
        T[low_value_flags] = 0
        T = (T+T.T)/2 #Forcing symmetry
    return T,t

 
def QR_alg_shifted(T):
    """Function to implement QR algorithm with wilkinson shift to 
    find eigen values"""
    m,_ = np.shape(T)
    t = []
    t.append(np.absolute(T[m-1,m-2]))
    while np.absolute(T[m-1,m-2]) >= 1e-12:
        mu = wilkinson_shift(T)
        Tmu = T - mu*np.identity(m)
        Q,R = np.linalg.qr(Tmu)
        T = np.dot(R,Q) + mu*np.identity(m)
        t.append(np.absolute(T[m-1,m-2]))
        low_value_flags = np.absolute(T) < 1e-12
        T[low_value_flags] = 0 #rounding off insiginificant values to zero
        T = (T+T.T)/2 #Forcing symmetry
    return T,t

def QR_alg_driver(A, shift=False):
    """Driver function to return eigen values, given a square symmetric matrix.
    Eigen values are returned as an array and also returns T[m-1,m-1] element
    of each submatrix undergoing QR algorithm"""
    m,_ = np.shape(A)
    T = tridiag(A)
    eigen_values = []
    all_t_temp = []
    for i in range(m-1,0,-1):
        if shift == True:
            T_new,t = QR_alg_shifted(T)
        else:
            T_new,t = QR_alg(T)
        m_new,_ = np.shape(T_new)
        eigen_values.append(T_new[-1,-1])
        all_t_temp.append(t)
        T = T_new[0:m_new-1,0:m_new-1]   
    eigen_values.append(T[0,0])
    all_t = sum(all_t_temp,[])
    return eigen_values,all_t
def plotting_function():
    A = hilbert(4)
    _,all_t_shift = QR_alg_driver(A, shift=True)
    _,all_t_noshift = QR_alg_driver(A, shift=False)
    plot_list1 = []
    plot_list2 = []
    for i in all_t_shift:
        plot_list1 = plot_list1 + i
    for j in all_t_noshift:
        plot_list2 = plot_list2 + j
    #fig, ax = plt.subplots(nrows=1, ncols=2)
    fig, ax = plt.subplots(2,figsize=(10,7))
    ys1 = [j for j in plot_list1]
    xs1 = [i for i in range(0,len(plot_list1))]
    ys2 = [j for j in plot_list2]
    xs2 = [i for i in range(0,len(plot_list2))]
    ax[0].semilogy(xs1,ys1)
    ax[0].set_title("With shift")
    ax[1].semilogy(xs2,ys2)
    ax[1].set_title("Without shift")
    fig.subplots_adjust(hspace=0.3)
    
if __name__ == "__main__":
    matrices = {
        "hilbert": hilbert(4),
        "diag(1,2,3,4)+ones": np.diag([1, 2, 3, 4]) + np.ones((4, 4)),
        "diag(5,6,7,8)+ones": np.diag([5, 6, 7, 8]) + np.ones((4, 4)),
    }

    fig, ax = plt.subplots(len(matrices.keys()), 2, figsize=(10, 10))

    for i, (mat, A) in enumerate(matrices.items()):
        print(f"A = {mat}")
        Λ,_ = np.linalg.eig(A)
        print(f"Λ = {np.sort(Λ)}\n")
        for j, shift in enumerate([True, False]):
            Λ, conv = QR_alg_driver(A.copy(), shift)
            ax[i, j].semilogy(range(len(conv)), conv, ".-")
            ax[i, j].set_title(f"A = {mat}, shift = {shift}")

    plt.show()    
 

    
    