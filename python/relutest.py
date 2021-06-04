import time
from algopy import UTPM
import tensorflow as tf
import numpy as np
from tensorflow import keras
import alm
from matplotlib import pyplot

W,N = 8,40 
delta = 0.1
rng = np.random.default_rng()

x = np.linspace(0,np.pi,N).reshape((N,1))
#y = -np.sin(.8*np.pi*x)+rng.normal(0,delta,x.shape)
y = np.sin(x*x)+rng.normal(0,delta,x.shape)

per = rng.permutation(N)


#sigma  = lambda x: tf.nn.relu(x)
#sigma_ = lambda x: np.greater(x,0,x)
sigma  = lambda x: tf.math.tanh(x)
sigma_ = lambda x: 2/(np.cosh(2*x)+1)

tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)

model = keras.Sequential()
model.add(keras.layers.Dense(activation="relu",units=W,input_shape=(x.shape[1],)))
model.add(keras.layers.Dense(activation="relu",units=W,input_shape=(x.shape[1],)))
model.add(keras.layers.Dense(units=y.shape[1]))
sgd = keras.optimizers.Adam()
es = keras.callbacks.EarlyStopping(monitor='loss',patience=10)
model.compile(optimizer=sgd, loss='mean_squared_error')

malm = keras.Sequential()
malm.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W,input_shape=(x.shape[1],)))
malm.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W))
malm.add(alm.Dense_d(activation=tau,activation_=tau_,units=y.shape[1]))

w = model.get_weights()
malm.set_weights(w)

t0 = time.process_time()
#hist = model.fit(x[per,:],y[per,:],batch_size=N,epochs=4000,verbose=0)
#print(len(hist.history['loss']),hist.history['loss'][-1])
#print(time.process_time()-t0)
xsm = np.linspace(0,np.pi,1000).reshape((1000,1))
#ysm = model(xsm)


almnet = alm.ALMModel(malm, x[per,:], y[per,:])
hist2 = almnet.fit_alm()

pyplot.figure(1)
p1, = pyplot.plot(x,y,'k+')
p2, = pyplot.plot(xsm,almnet.model(xsm),'r-')
pyplot.xlabel("x")
pyplot.ylabel("y")
pyplot.legend([p1,p2],['Training Data',"Network prediction"],loc="lower left")
pyplot.gcf().subplots_adjust(bottom=0.2,left=0.2)
pyplot.figure(2)
pyplot.semilogy(range(len(hist2['loss'])),hist2['loss'])
pyplot.xlabel("epoch")
pyplot.ylabel("MSE")
pyplot.gcf().subplots_adjust(bottom=0.2,left=0.2)
pyplot.show()
