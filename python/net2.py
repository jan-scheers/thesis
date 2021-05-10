import numpy as np
import tensorflow as tf
from tensorflow import keras

class Dense_d(keras.layers.Dense):
    def __init__(self,activation_,**kwargs):
        super().__init__(**kwargs)
        self.activation_ = activation_

class ALMModel:
    def __init__(self,model,x,y):
        self.model = model
        self.x = x
        self.y = y
        self.batch_size = x.shape[0]
        self.input_shape = x.shape[1:] 
        self.z = self.pred_state()
        print(self.z)

    def pred_state(self):
        z = [self.x]
        for i in range(0,len(self.model.layers)):
            z.append(self.model.layers[i](z[i]))
        return z


    def JL(self):
        layers = self.model.layers

#        w,b = layers[0].get_weights()
#        w = np.r_[w,b[np.newaxis,:]].transpose()
#        x = np.c_[self.x,np.ones((self.batch_size,1))].transpose()
#        w_ = w.dot(x)
#        w_ = -x*w_[:,np.newaxis,:]
#        w_ = w_*np.eye(w.shape[0])[:,:,np.newaxis,np.newaxis]
#        w_ = np.swapaxes(w_,1,2).reshape(w.shape[1]*w.shape[0],w.shape[0]*x.shape[1]).transpose()
        w_ = []
        np.set_printoptions(linewidth=200)
        for i in range(0, len(layers)):
            wi,b = layers[i].get_weights()
            wi = np.r_[wi,b[np.newaxis,:]].transpose()
            zi = np.c_[self.z[i],np.ones((self.batch_size,1))].transpose()
            wi_ = layers[i].activation_(wi.dot(zi))
            wi_ = -zi*wi_[:,np.newaxis,:]
            wi_ = wi_*np.eye(wi.shape[0])[:,:,np.newaxis,np.newaxis]
            wi_ = np.swapaxes(wi_,1,2).reshape(wi.size,wi.shape[0]*self.batch_size).transpose()
            w_.append(wi_)
            print(wi_)
        



