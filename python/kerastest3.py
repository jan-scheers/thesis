import net
from algopy import UTPM
import tensorflow as tf
import numpy as np
from tensorflow import keras
import alm
from matplotlib import pyplot

I,O,W,D,N, = 1,1,8,2,21
mu = 10

shape = (W,D,I,O)
sigma  = lambda x: tf.nn.relu(x)
sigma_ = lambda x: np.greater(x,0,x)

tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)

x = np.linspace(0,1,N).reshape((1,N))
y = -np.sin(.8*np.pi*x)+np.random.normal(0,0.1,x.shape)
#y = np.arange(O*N).reshape((O,N))
#x = np.random.normal(0,1,I*N).reshape((I,N))
#y = np.random.normal(0,1,O*N).reshape((O,N))
#u = np.random.random_sample((W*W*(D-1) + D*W + I*W + O*W + O + D*W*N,))
#l = np.random.random_sample((D*W*N,))

model = keras.Sequential()
model.add(alm.Dense_d(activation_=sigma_,units=W,activation=sigma,input_shape=(I,)))
model.add(alm.Dense_d(activation_=sigma_,units=W,activation=sigma))
model.add(alm.Dense_d(activation_=tau_,units=O,activation=tau))

nnet = alm.ALMModel(model, x.transpose(), y.transpose())
model.summary()
#print(nnet.nz,l.size)

#J2 = nnet.JL(u, mu).toarray()

#u = np.arange(u.size)
#w,z = nnet.read(u)
#nnet.set_weights(w)

#print(nnet.model.get_weights())

#print(nnet.L(u,1,np.zeros(l.size)))

u = nnet.training_loop_alm()

pyplot.plot(nnet.x,nnet.y,'k+',nnet.x,nnet.model(nnet.x),'r-')
pyplot.show()
