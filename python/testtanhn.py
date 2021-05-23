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


for N in [10,20,40]:
    x = np.linspace(0,np.pi,N).reshape((N,1))
    y = np.sin(x*x)+rng.normal(0,delta,x.shape)
    i = rng.permutation(N)
    i_train,i_val = i[:int(.8*N)],i[int(.8*N):]


    te = np.empty((10,4))
    for k in range(10):
        model = keras.Sequential()
        model.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W,input_shape=(x.shape[1],)))
        model.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W))
        model.add(alm.Dense_d(activation=tau,activation_=tau_,units=y.shape[1]))
        sgd = keras.optimizers.Adam(learning_rate=.01)
        es = keras.callbacks.EarlyStopping(monitor='loss',patience=10)
        model.compile(optimizer=sgd, loss='mean_squared_error')
    
        w = model.get_weights()
        
        t0 = time.process_time()
        hist = model.fit(x[i_train,:],y[i_train,:],batch_size=i_train.size,validation_data=(x[i_val,:],y[i_val,:]),epochs=2000,callbacks=[es],verbose=0)

        t1 = time.process_time()-t0
        e1 = len(hist.history['loss'])


        model.set_weights(w)
        almnet = alm.ALMModel(model, x[i_train,:], y[i_train,:])

        t0 = time.process_time()
        hist2 = almnet.fit_alm(val_data=(x[i_val,:],y[i_val,:]))
        t2 = time.process_time()-t0
        e2 = hist2['loss'].size
    
        te[k,:] = [t1,t2,e1,e2]
        print(te[k,:])
        with open('testtanhn.npy','ab') as f:
            np.save(f,np.concatenate([np.array(value) for value in hist.history.values()]))
            np.save(f,np.concatenate([np.array(value) for value in hist2.values()]))
    with open('testtanhn.npy','ab') as f:
        np.save(f,te)

   

    
