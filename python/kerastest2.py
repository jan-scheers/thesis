import net
from algopy import UTPM
import tensorflow as tf
import numpy as np
from tensorflow import keras
import alm
from matplotlib import pyplot

W,N = 8,20
delta = 0.1
rng = np.random.default_rng()

x = np.linspace(0,np.pi,N).reshape((N,1))
#y = -np.sin(.8*np.pi*x)+rng.normal(0,delta,x.shape)
y = np.sin(x*x)+rng.normal(0,delta,x.shape)

i = rng.permutation(N)
i_train,i_val = i[:int(.8*N)],i[int(.8*N):]

sigma  = lambda x: tf.math.tanh(x)
sigma_ = lambda x: 2/(np.cosh(2*x)+1)

tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)



model = keras.Sequential()
model.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W,input_shape=(x.shape[1],)))
model.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W))
model.add(alm.Dense_d(activation=tau,activation_=tau_,units=y.shape[1]))
sgd = keras.optimizers.Adam(learning_rate=.01)
es = keras.callbacks.EarlyStopping(monitor='loss',patience=10)
model.compile(optimizer=sgd, loss='mean_squared_error')

w = model.get_weights()

hist = model.fit(x[i_train,:],y[i_train,:],batch_size=i_train.size,validation_data=(x[i_val,:],y[i_val,:]),epochs=2000,callbacks=[es],verbose=0)
print(len(hist.history['loss']),hist.history['loss'][-1])
print(hist.history.keys())

x_sm = np.linspace(0,np.pi,1000).reshape((1000,1))
y_adam = model(x_sm)

model.set_weights(w)

almnet = alm.ALMModel(model, x[i_train,:], y[i_train,:])
hist2 = almnet.fit_alm(val_data=(x[i_val,:],y[i_val,:]))


pyplot.plot(x,y,'k+',x_sm,y_adam,'b-',x_sm,almnet.model(x_sm),'r-')

#epochs = np.arange(hist2['tol'].size)
#pyplot.semilogy(epochs,hist2['tol'],epochs,hist2['loss'],epochs,hist2['val_loss'])
pyplot.show()
