import net
from algopy import UTPM
import tensorflow as tf
import numpy as np
from tensorflow import keras
import alm
from matplotlib import pyplot

I,O,W,D,N, = 2,2,3,2,7
mu = 10

shape = (W,D,I,O)
sigma  = lambda x: np.tanh(x)
sigma_ = lambda x: 2/(np.cosh(2*x)+1)

tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)

x = np.random.normal(0,1,I*N).reshape((I,N))
y = np.random.normal(0,1,O*N).reshape((O,N))
u = np.random.random_sample((W*W*(D-1) + D*W + I*W + O*W + O + D*W*N,))
l = np.random.random_sample((D*W*N,))

nn = net.Net(shape,sigma,sigma_,tau,tau_,x,y)

_,z = nn.sim(u)
n_u = u.size
u[n_u-W*D*N:n_u] = z.ravel()

L = nn.eval_L(u,mu,l)
J = nn.eval_J_L(u,mu,l)

u_U = UTPM.init_jacobian(u)
L_U = net.eval_L_U(u_U,mu,l,nn)
J_U = UTPM.extract_jacobian(L_U)
J_U = J_U.reshape(J_U.shape[1:3])


model = keras.Sequential()
model.add(alm.Dense_d(activation_=sigma_,units=W,activation=tf.math.tanh,input_shape=(I,)))
model.add(alm.Dense_d(activation_=sigma_,units=W,activation=tf.math.tanh))
model.add(alm.Dense_d(activation_=tau_,units=O,activation=tau))
n2 = alm.ALMModel(model, x.transpose(), y.transpose())
model.summary()


g = np.arange(W)+1
a = g*(I+1)
b = a[2] + g*(W+1)
c = b[2] + W+1

g = np.r_[a,b,c]-1
h = np.arange(n_u-D*W*N,n_u)

#mask = np.ones((n_u,),bool)

#mask[g] = 0
#mask[h] = 0

#k = np.arange(n_u)
t = np.arange(n_u)
#t[mask] = k[0:h[0]-D*W-O]
#t[g] = k[h[0]-D*W-O:h[0]]
#t[h] = k[h]

L2 = n2.L(u[t], mu, l)
J2 = n2.JL(u[t], mu).toarray()

Jnz = J2 != 0

#J2 = np.c_[J2[:,mask],J2[:,g],J2[:,h]]

#print(np.allclose(L,np.ravel(L_U.data[0,0])))
#print(np.allclose(L,L2))
#print(np.allclose(J,J_U))
#print(np.allclose(J,J2))

#Jnz = J != 0

Jnz2 = np.r_[Jnz[-O*N:,:],Jnz[:-O*N,:]]

np.set_printoptions(precision=3,linewidth=200)
pyplot.figure()
pyplot.matshow(Jnz,cmap="binary")
pyplot.figure()
pyplot.matshow(Jnz2,cmap="binary")
pyplot.show()

