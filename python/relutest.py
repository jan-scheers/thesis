import net
from algopy import UTPM
import tensorflow as tf
import numpy as np
from tensorflow import keras
import alm
from matplotlib import pyplot

W,N = 20,200
delta = 0.1
rng = np.random.default_rng()

x = np.linspace(0,1.5*np.pi,N).reshape((N,1))
#y = -np.sin(.8*np.pi*x)+rng.normal(0,delta,x.shape)
y = np.sin(x*x)#+rng.normal(0,delta,x.shape)

i = rng.permutation(N)
i_train,i_val = i[:int(.8*N)],i[int(.8*N):]


#sigma  = lambda x: tf.nn.relu(x)
#sigma_ = lambda x: np.greater(x,0,x)
sigma  = lambda x: tf.math.tanh(x)
sigma_ = lambda x: 2/(np.cosh(2*x)+1)

tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)

model = keras.Sequential()
model.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W,input_shape=(x.shape[1],)))
model.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W))
model.add(alm.Dense_d(activation=tau,activation_=tau_,units=y.shape[1]))
sgd = keras.optimizers.Adam()
es = keras.callbacks.EarlyStopping(monitor='loss',patience=10)
model.compile(optimizer=sgd, loss='mean_squared_error')

w = model.get_weights()

hist = model.fit(x[i_train,:],y[i_train,:],batch_size=i_train.size,validation_data=(x[i_val,:],y[i_val,:]),epochs=4000,callbacks=[es],verbose=0)
print(len(hist.history['loss']),hist.history['loss'][-1])

model.set_weights(w)

almnet = alm.ALMModel(model, x[i_train,:], y[i_train,:])
hist2 = almnet.fit_alm(val_data=(x[i_val,:],y[i_val,:]))

pyplot.plot(x,y,'k+',x,almnet.model(x),'r-')
#print(hist.history) 
pyplot.show()
