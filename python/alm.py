import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import sparse
import scipy.optimize as op
import scipy.linalg as la

class Dense_d(keras.layers.Dense):
    def __init__(self,activation_,**kwargs):
        super().__init__(**kwargs)
        self.activation_ = activation_

class ALMModel:
    def __init__(self,model,x,y):
        self.model = model
        t = self.model(x)
        self.x = x
        self.y = y
        self.batch_size = x.shape[0]
        self.nz = sum([self.batch_size*model.layers[i].units for i in range(len(model.layers)-1)])

    def read(self,u):
        layers = self.model.layers
        w = []
        p = 0
        for i in range(0,len(layers)):
            s = layers[i].weights[0].shape
            m,n = s[1],s[0]+1
            w.append(u[p:p+m*n].reshape(m,n))
            p = p+m*n

        z = [self.x.transpose()]
        for i in range(1,len(layers)):
            m,n = w[i].shape[1]-1,self.batch_size
            zi = u[p:p+m*n].reshape(m,n)
            z.append(zi)
            p = p+m*n
        y = np.r_[self.y.transpose()]
        z.append(y)

        return w,z
    
    def write(self):
        u = np.empty(0)
        for layer in self.model.layers:
            w,b = layer.weights
            m,n = w.shape[1],w.shape[0]+1
            w = np.c_[w.numpy().transpose(),b.numpy().reshape(m,1)]
            u = np.append(u,w)
        z = [self.x]
        for i,layer in enumerate(self.model.layers):
            z.append(layer(z[i]))

        z = z[1:-1]
        z = [zi.numpy().transpose() for zi in z] 
        u = np.append(u,np.concatenate(z))
        return u

    def set_weights(self,w):
        for i,layer in enumerate(self.model.layers):
            wi = w[i].transpose()
            layer.set_weights([wi[:-1,:],wi[-1:,].reshape((-1,))])

    def h(self,u):
        w,z = self.read(u)
        h = []
        for i,layer in enumerate(self.model.layers[:-1]):
            zi_e = np.r_[z[i],np.ones((1,self.batch_size))]
            hi = z[i+1] - np.array(layer.activation(w[i].dot(zi_e)))
            h.append(hi.reshape(-1))
        return np.concatenate(h)

    def L(self,u,beta,l):
        w,z = self.read(u)
        
        out_layer = self.model.layers[-1]
        z_e = np.r_[z[-2],np.ones((1,self.batch_size))]
        F = z[-1] - np.array(out_layer.activation(w[-1].dot(z_e)))
        F = F.reshape(-1)/np.sqrt(beta)

        h = self.h(u)+l/beta
        return np.concatenate([h,F])

        
    def JL(self,u,beta):
        layers = self.model.layers
        w,z = self.read(u)

        w_ = []
        z_ = []
        np.set_printoptions(linewidth=200)
        for i in range(len(layers)):
            # Add bias to weights
            zi_e = np.r_[z[i],np.ones((1,self.batch_size))]

            ai = layers[i].activation_(w[i].dot(zi_e))

            wi_ = -zi_e*ai[:,np.newaxis,:]
            wi_ = wi_*np.eye(w[i].shape[0])[:,:,np.newaxis,np.newaxis]
            wi_ = np.swapaxes(wi_,1,2).reshape(w[i].size,w[i].shape[0]*self.batch_size).transpose()

            w_.append(wi_)

            if i > 0:
                zi_ = -w[i][:,:-1].transpose()[:,:,np.newaxis]*ai
                zi_ = np.eye(self.batch_size)*zi_[:,:,:,np.newaxis]
                zi_ = np.swapaxes(zi_,1,2).reshape(z[i].size,z[i+1].size).transpose()
                z_.append(zi_)

        
        w_[-1] = w_[-1]/np.sqrt(beta)
        z_[-1] = z_[-1]/np.sqrt(beta)

        w_ = [sparse.csc_matrix(wi_) for wi_ in w_]
        z_ = [sparse.csc_matrix(zi_) for zi_ in z_]
        
        z_ = sparse.block_diag(z_)
        z_ = sparse.vstack((sparse.csr_matrix((w_[0].shape[0],z_.shape[1])),z_))
        z_ = z_+sparse.eye(z_.shape[0],z_.shape[1])
        w_ = sparse.block_diag(w_)
        J = sparse.hstack((w_,z_))
        return J
            

    def fit_alm(self,val_data=None,beta=10,tau=1e-2):
        l = np.random.normal(0,1,self.nz)
        u = self.write()
        sigma_0,h_0 = 1,la.norm(self.h(u))
        hist = {"loss":np.empty(0),"tol":np.empty(0),"njev":np.empty(0)}
        if(val_data):
            hist['val_loss'] = np.empty(0)

        for k in range(12):
            beta_k = np.power(beta,k)
            eta_k = 1/beta_k
            fun = lambda u: self.L(u,beta_k,l)
            jac = lambda u: self.JL(u,beta_k)
            try:
                sol = op.least_squares(fun,u,jac,ftol=None,xtol=None,gtol=eta_k,tr_solver='lsmr')
            except:
                print("Divide by 0")
                break
            u = sol.x

            w,_ = self.read(u)
            self.set_weights(w)

            h = self.h(u)

            sigma = sigma_0*np.amin([h_0*np.power(np.log(2),2)/la.norm(h)/(k+1)/np.power(np.log(k+2),2),1])
            l = l + sigma*h

            # Convergence of Lagrangian function
            jac = self.JL(u,beta_k).toarray()
            tol = la.norm(2*self.JL(u,beta_k).transpose()*self.L(u,beta_k,l))+la.norm(h)
            hist['tol'] = np.append(hist['tol'],tol)
            hist['njev'] = np.append(hist['njev'],sol.njev)

            # Convergence of training loss
            y_pred = self.model(self.x)
            mse = keras.losses.MeanSquaredError()
            loss = mse(self.y,y_pred).numpy()
            hist['loss'] = np.append(hist['loss'],loss)
            if(val_data):
                # Convergence of val/test loss
                y_val_pred = self.model(val_data[0])
                val_loss = mse(y_val_pred,val_data[1]).numpy()
                hist['val_loss'] = np.append(hist['val_loss'],val_loss)
                print("epoch: ",k+1,"DL: ",tol,"njev: ",sol.njev, "loss = ",loss,"val_loss = ",val_loss)
            else:
                print("epoch: ",k+1,"DL: ",tol,"njev: ",sol.njev, "loss = ",loss)

            if k>1 and ((1+tau)*hist['loss'][-1] > hist['loss'][-2]):
                break
        return hist
            
            



