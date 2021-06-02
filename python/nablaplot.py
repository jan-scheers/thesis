import net
from algopy import UTPM
import tensorflow as tf
import numpy as np
from tensorflow import keras
import alm
from matplotlib import pyplot
import time

rng = np.random.default_rng()
delta = 0.1
W = 8

sigma  = lambda x: tf.math.tanh(x)
sigma_ = lambda x: 2/(np.cosh(2*x)+1)

tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)


N = 40
x = np.linspace(0,np.pi,N).reshape((N,1))
y = np.sin(x*x)+rng.normal(0,delta,x.shape)
i = rng.permutation(N)
i_train,i_val = i[:int(.8*N)],i[int(.8*N):]

model = keras.Sequential()
model.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W,input_shape=(x.shape[1],)))
model.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W))
model.add(alm.Dense_d(activation=tau,activation_=tau_,units=y.shape[1]))
sgd = keras.optimizers.Adam(learning_rate=.01)
es = keras.callbacks.EarlyStopping(monitor='loss',patience=10)
model.compile(optimizer=sgd, loss='mean_squared_error')
    
w = model.get_weights()

almnet = alm.ALMModel(model, x[i_train,:], y[i_train,:])

hist = almnet.fit_alm(val_data=(x[i_val,:],y[i_val,:]))


p1, = pyplot.semilogy(np.arange(10)+1,hist['tol'],"b-",linewidth=2)
p2, = pyplot.semilogy(np.arange(10)+1,hist['loss'],'r-',linewidth=2)
pyplot.grid(True)
pyplot.xlabel("epoch")
pyplot.legend([p1,p2],[r'$\nabla_u\mathcal{L}_\beta(u_k,\lambda_k)$',"MSE training loss"])

pyplot.show()

with open('nabla.npy','ab') as f:
    np.save(hist)



   

    
