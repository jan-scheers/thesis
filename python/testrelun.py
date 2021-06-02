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
W = 16
K = 20

sigma  = lambda x: tf.nn.relu(x)
sigma_ = lambda x: np.greater(x,0,x)

tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)


for N in [20,40,80]:
    x = np.linspace(0,np.pi,N).reshape((N,1))
    y = np.sin(x*x)+rng.normal(0,delta,x.shape)
    per = rng.permutation(N)

    te = np.empty((K,4))
    for k in range(K):
        madam = keras.Sequential() 
        madam.add(keras.layers.Dense(activation="relu",units=W,input_shape=(x.shape[1],)))
        madam.add(keras.layers.Dense(activation="relu",units=W))
        madam.add(keras.layers.Dense(units=y.shape[1]))
        adam = keras.optimizers.Adam()
        es = keras.callbacks.EarlyStopping(monitor='loss',patience=10)
        madam.compile(optimizer=adam, loss='mean_squared_error')

        malm = keras.Sequential()
        malm.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W,input_shape=(x.shape[1],)))
        malm.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W))
        malm.add(alm.Dense_d(activation=tau,activation_=tau_,units=y.shape[1]))
    
        w = madam.get_weights()
        malm.set_weights(w)
        
        t0 = time.process_time()
        hist = madam.fit(x[per,:],y[per,:],batch_size=N,epochs=4000,callbacks=[es],verbose=0)
        t1 = time.process_time()-t0
        e1 = len(hist.history['loss'])
        print("t: ",t1,"k: ",e1,'loss: ', hist.history['loss'][-1])

        almnet = alm.ALMModel(malm, x[per,:], y[per,:])

        t0 = time.process_time()
        hist2 = almnet.fit_alm()
        t2 = time.process_time()-t0
        e2 = hist2['loss'].size
        print("t: ",t2,"k: ",e2, 'loss: ', hist2['loss'][-1])
    
        te[k,:] = [t1,t2,e1,e2]
        with open('testrelun.npy','ab') as f:
            np.save(f,np.array(hist.history['loss']))
            np.save(f,np.concatenate([np.array(value) for value in hist2.values()]).reshape((3,-1)).transpose())
    with open('testrelun.npy','ab') as f:
        np.save(f,te)

   

    
