import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import sparse

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

    def pred_state(self):
        z = [self.x]
        for i in range(0,len(self.model.layers)):
            z.append(np.array(self.model.layers[i](z[i])))
        return z
    
    def read(self,u):
        layers = self.model.layers
        w = []
        p = 0
        for i in range(0,len(layers)):
            s = layers[i].weights[0].shape
            m,n = s[1],s[0]+1
            w.append(u[p:p+m*n].reshape(m,n))
            p = p+m*n

        z = []
        for i in range(1,len(layers)):
            m,n = w[i].shape[1]-1,self.batch_size
            z.append(u[p:p+m*n].reshape(m,n))
            p = p+m*n

        return w,z



    def JL(self,u,rho):
        layers = self.model.layers
        w,z = self.read(u)
        z.insert(0,self.x.transpose())
        z.append(self.y.transpose())

        w_ = []
        z_ = []
        np.set_printoptions(linewidth=200)
        for i in range(len(layers)):
            # Add bias to weights

            zi = np.r_[z[i],np.ones((1,self.batch_size))]


            ai = layers[i].activation_(w[i].dot(zi))

            wi_ = -zi*ai[:,np.newaxis,:]
            wi_ = wi_*np.eye(w[i].shape[0])[:,:,np.newaxis,np.newaxis]
            wi_ = np.swapaxes(wi_,1,2).reshape(w[i].size,w[i].shape[0]*self.batch_size).transpose()

            w_.append(wi_)

            if i > 0:
                zi_ = -w[i][:,:-1].transpose()[:,:,np.newaxis]*ai
                zi_ = np.eye(self.batch_size)*zi_[:,:,:,np.newaxis]
                zi_ = np.swapaxes(zi_,1,2).reshape(z[i].size,z[i+1].size).transpose()
                z_.append(zi_)

        
        w_[-1] = w_[-1]/np.sqrt(rho)
        z_[-1] = z_[-1]/np.sqrt(rho)

        w_ = [sparse.csc_matrix(wi_) for wi_ in w_]
        z_ = [sparse.csc_matrix(zi_) for zi_ in z_]
        
        z_ = sparse.block_diag(z_)
        z_ = sparse.vstack((sparse.csr_matrix((w_[0].shape[0],z_.shape[1])),z_))
        z_ = z_+sparse.eye(z_.shape[0],z_.shape[1])
        w_ = sparse.block_diag(w_)
        J = sparse.hstack((w_,z_))
        return J


            

