""" Augmented Lagrangian method """
import numpy as np
from scipy import sparse as sp
from scipy import linalg as la 
from scipy import optimize as op
import sys
from algopy import UTPM
import matplotlib.pyplot as pyplot


def sigma(x):
    """ Activation function """
    return UTPM.tanh(x)

def sigma_(x):
    return 2/(np.cosh(2*x)+1)

def extract_var(u,B,N):
    p,n = 0,B
    w1 = u[p:n].reshape((B,1))
    p = n
    n = p + B*B
    w2 = u[p:n].reshape((B,B))
    p = n
    n = p + B
    w3 = u[p:n].reshape((1,B))
    p = n
    n = p + B
    b1 = u[p:n].reshape((B,1))
    p = n
    n = p + B
    b2 = u[p:n].reshape((B,1))
    p = n
    n = p + 1
    b3 = u[p:n].reshape((1,1))
    p = n
    n = p + B*N
    z1 = u[p:n].reshape((B,N))
    p = n
    n = p + B*N
    z2 = u[p:n].reshape((B,N))
    p = n
    n = p + 2*B*N
    l = u[p:n].reshape((1,2*B*N))

    return w1,w2,w3,b1,b2,b3,z1,z2,l

def eval_L_U(u,x,y,c,B,N):
    """ Multilevel perceptron """
    w1,w2,w3,b1,b2,b3,z1,z2,l = extract_var(u,B,N)
    F  = y  -      (UTPM.dot(w3,z2) + b3)
    h1 = z1 - sigma(UTPM.dot(w1,x)  + b1)
    h2 = z2 - sigma(UTPM.dot(w2,z1) + b2)
    F,h1,h2 = F.reshape((1,-1))/np.sqrt(c),h1.reshape((1,-1)),h2.reshape((1,-1))

    h = UTPM(np.concatenate([h1.data,h2.data],3))
    h = h+l/c
    print(h.shape)
    return UTPM(np.concatenate([F.data,h.data],3))

def eval_L(u,x,y,c,B,N):
    """ Multilevel perceptron """
    w1,w2,w3,b1,b2,b3,z1,z2,l = extract_var(u,B,N)
    F  = y  -      (w3.dot(z2) + b3)
    h1 = z1 - np.tanh(w1.dot(x)  + b1)
    h2 = z2 - np.tanh(w2.dot(z1) + b2)
    F,h1,h2 = F.reshape((1,-1))/np.sqrt(c),h1.reshape((1,-1)),h2.reshape((1,-1))

    h = np.concatenate([h1,h2],1)
    h = h+l/c
    return np.ravel(np.concatenate([F,h],1))
def eval_J_L(u,x,y,c,B,N):
    w1,w2,w3,b1,b2,b3,z1,z2,l = extract_var(u,B,N)
    n_u, = u.shape 
    J = np.zeros((2*B*N+N,n_u))
    F  = slice(0,N)
    h1 = slice(N,(B+1)*N)
    h2 = slice((B+1)*N,(2*B+1)*N)

    w1_ = -x*sigma_(w1.dot(x)+b1) 
    w1_ = np.eye(B)[:,np.newaxis,:]*w1_.transpose()
    w1_ = w1_.reshape(B*N,B)

    p = 0
    n = B
    J[h1,p:n] = w1_


    w2_ = sigma_(w2.dot(z1)+b2)
    w2_ = -z1*w2_[:,np.newaxis,:]
    w2_ = w2_*np.eye(B)[:,:,np.newaxis,np.newaxis]
    w2_ = np.swapaxes(w2_,1,2).reshape(B*B,B*N).transpose()

    p = n
    n = p+B*B
    J[h2,p:n] = w2_

    w3_ = -z2.transpose()/np.sqrt(c)

    p = n
    n = p+B
    J[F,p:n] = w3_


    b1_ = -sigma_(w1.dot(x)+b1)
    b1_ = np.eye(B)[:,np.newaxis,:]*b1_.transpose()
    b1_ = b1_.reshape(B*N,B)

    p = n
    n = p+B
    J[h1,p:n] = b1_

    b2_ = -sigma_(w2.dot(z1)+b2)
    b2_ = np.eye(B)[:,np.newaxis,:]*b2_.transpose()
    b2_ = b2_.reshape(B*N,B)

    p = n
    n = p+B
    J[h2,p:n] = b2_

    b3_ = -np.ones((N,1))/np.sqrt(c)

    p = n
    n = p+1
    J[F,p:n] = b3_

    z1_ = -w2.transpose()[:,:,np.newaxis]*sigma_(w2.dot(z1)+b2)
    z1_ = np.eye(N)*z1_[:,:,:,np.newaxis]
    z1_ = np.swapaxes(z1_,1,2).reshape(B*N,B*N).transpose()

    p = n
    n = p+B*N
    J[h1,p:n] = np.eye(B*N)
    J[h2,p:n] = z1_

    z2_ = -np.eye(N)*w3[:,:,np.newaxis,np.newaxis]/np.sqrt(c) 
    z2_ = np.swapaxes(z2_,1,2).reshape(N,B*N)

    p = n
    n = p+B*N
    J[h2,p:n] = np.eye(B*N)
    J[F,p:n] = z2_
    
    p = n
    n = p+2*B*N
    J[np.r_[h1,h2],p:n] = np.eye(2*B*N)/c

    return J

def sim(w1,w2,w3,b1,b2,b3,x):
    z1 = np.tanh(w1.dot(x) + b1)
    z2 = np.tanh(w2.dot(z1)+ b2)
    y = w3.dot(z2) + b3
    return np.ravel(y),np.ravel(z1),np.ravel(z2)

B, N = 3, 11

x = np.reshape(np.linspace(0, 1, N), [1, N])
y = -np.sin(.8*np.pi*x)
#x = np.random.random_sample([1,N])
#y = np.random.random_sample([1,N])
c = 7

w1 = np.random.random_sample([B, 1])
w2 = np.random.random_sample([B, B])
w3 = np.random.random_sample([1, B])
b1 = np.random.random_sample([B, 1])
b2 = np.random.random_sample([B, 1])
b3 = np.random.random_sample([1, 1])
_,z1,z2 = sim(w1,w2,w3,b1,b2,b3,x)
l =  np.random.random_sample([B*N*2, 1])

u = np.concatenate([w1,w2,w3,b1,b2,b3,z1,z2,l],None)
L = eval_L(u,x,y,c,B,N)
J_L = eval_J_L(u,x,y,c,B,N)
u = UTPM.init_jacobian(u)

L_U = eval_L_U(u,x,y,c,B,N)
J = UTPM.extract_jacobian(L_U)
J = J.reshape(J.shape[1:3])

np.set_printoptions(linewidth=300,precision=0,threshold=sys.maxsize)

print(np.allclose(J_L,J))
print(np.allclose(L,np.ravel(L_U.data[0,0])))

c = 2
#for i = 1:10:
#    fun = lambda u:eval_L(u,x,y,c,B,N)
#    jac = lambda u:eval_J_L(u,x,y,c,B,N)
#    u = np.concatenate([w1,w2,w3,b1,b2,b3,z1,z2,l],None)
#    sol = op.least_squares(fun, u, jac)
#    w1,w2,w3,b1,b2,b3,z1,z2,_ = extract_var(sol.x,B,N)
#    ys,z1s,z2s = sol(w1,w2,w3,b1,b2,b3,x)
#    h = [z1-z1s;z2-z2s]
#    l_next = l + c*h

#    u 
#print(sol.x)
