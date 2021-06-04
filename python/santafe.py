import numpy as np
from scipy import linalg
import alm
import tensorflow as tf
from tensorflow import keras
import time

class cb100(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 200 == 0:
            print("epoch: ",epoch,", loss: ",logs['loss'])
rng = np.random.default_rng()


sigma  = lambda x: tf.math.tanh(x)
sigma_ = lambda x: 2/(np.cosh(2*x)+1)
tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)

lasertrain = np.genfromtxt('lasertrain.csv')
lasertest  = np.genfromtxt('laserpred.csv')
lm,ls = np.mean(lasertrain),np.std(lasertrain)
lasertrain = (lasertrain-lm)/ls
lasertest = (lasertest-lm)/ls


lag = 80
Ntr = 600
W = 48
lasermat = linalg.hankel(lasertrain,lasertrain[0:lag+1])[:-lag,:]

te = np.empty((4,6))
for k in range(4):
    per = rng.permutation(lasertrain.size-lag)
    
    lasertrn = lasermat[per[:Ntr],:]
    laserval = lasermat[per[Ntr:],:]
    
    lasertrny = lasertrn[:,-1].reshape(-1,1)
    lasertrnx = lasertrn[:,:-1]
    
    laservaly = laserval[:,-1].reshape(-1,1)
    laservalx = laserval[:,:-1]
    
    madam = keras.Sequential()
    madam.add(keras.layers.Dense(activation="tanh",units=W,input_shape=(lag,)))
    madam.add(keras.layers.Dense(units=1))
    adam = keras.optimizers.Adam()
    madam.compile(optimizer=adam, loss='mean_squared_error')
    
    w = madam.get_weights()
    es = keras.callbacks.EarlyStopping(monitor='loss',patience=10)
    cb = cb100()
    t0 = time.process_time()
    hist = madam.fit(lasertrnx,lasertrny,batch_size=Ntr,epochs=10000,callbacks=[es,cb],validation_data=(laservalx,laservaly),validation_freq=200,verbose=0)
    t1 = time.process_time()-t0
    e1 = len(hist.history['loss'])
    print("t: ",t1,"k: ",e1,
          'loss: ', hist.history['loss'][-1],
      'val_loss: ', hist.history['val_loss'][-1])
    
    malm = keras.Sequential()
    malm.add(alm.Dense_d(activation=sigma,activation_=sigma_,units=W,input_shape=(lag,)))
    malm.add(alm.Dense_d(activation=tau,activation_=tau_,units=1))
    malm.set_weights(w)
    
    almnet = alm.ALMModel(malm,lasertrnx,lasertrny)
    t0 = time.process_time()
    hist2 = almnet.fit_alm(val_data=(laservalx,laservaly),tau=0)
    t2 = time.process_time()-t0
    e2 = hist2['loss'].size
    print("t: ",t2,"k: ",e2, 
          'loss: ', hist2['loss'][-1],
      'val_loss: ', hist2['val_loss'][-1])
    
    Ntest = lasertest.size
    adampred = lasertrain[-lag:]
    almpred = lasertrain[-lag:]
    for n in range(Ntest):
        adampred = np.append(adampred,
                       madam(adampred[-lag:].reshape(1,-1)))
        almpred = np.append(almpred,
                       malm(almpred[-lag:].reshape(1,-1)))
    adampred = adampred[lag:]
    almpred = almpred[lag:]

    mse = keras.losses.MeanSquaredError()
    t1_loss = mse(lasertest,adampred).numpy()
    t2_loss = mse(lasertest,almpred).numpy()
    print("ADAM test: ",t1_loss,"ALM test: ",t2_loss)
    te[k,:] = [t1,t2,e1,e2,t1_loss,t2_loss]
    with open('santafe.npy','ab') as f:
        np.save(f,np.array(hist.history['loss']))
        np.save(f,np.array(hist.history['val_loss']))
        np.save(f,np.concatenate([np.array(value) for value in hist2.values()]).reshape((4,-    1)).transpose())
        np.save(f,np.r_[adampred,almpred])
with open('santafe.npy','ab') as f:
    np.save(f,te)


