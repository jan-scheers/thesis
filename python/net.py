import numpy as np
from algopy import UTPM


class Net:
    # TODO: switch W,D? 
    def __init__(self,shape, sigma, sigma_, tau , tau_,x, y):
        """sigma = activation function, sigma_ = first deriv of sigma
           tau   = output function, tau_ = first deriv of tau
           shape[0] = W
           shape[1] = D
           shape[2] = I
           shape[3] = O """

        self.shape = shape
        self.sigma  = sigma
        self.sigma_ = sigma_
        self.tau  = tau
        self.tau_ = tau_
        self.x = x
        self.y = y
        self.W = shape[0]
        self.D = shape[1]
        self.I = shape[2]
        self.O = shape[3]
        self.N = x.shape[1]

    
    def extract(self,u):
        """ Extracts variables from variable vector """
        W,D,I,O,N = self.W,self.D,self.I,self.O,self.N
        
        # Input weights W0 = WxI
        p,n = 0,W*I
        w0 = u[p:n].reshape((W,I))
        p = n

        # Hidden weights  Wi = WxW
        w = []
        for i in range(D-1):
            n = p + W*W
            w.append(u[p:n].reshape((W,W)))
            p = n

        # Output weights WD = OxW
        n = p + O*W
        wd =  u[p:n].reshape((O,W))
        p = n
        
        # Hidden biases bi = Wx1
        b = []
        for i in range(D):
            n = p + W
            b.append(u[p:n].reshape((W,1)))
            p = n

        # Output bias bi = Ox1
        n = p + O
        bd = u[p:n].reshape((O,1))
        p = n
        
        # State vectors zi = WxN
        z = []
        for i in range(D):
            n = p + W*N
            z.append(u[p:n].reshape((W,N)))
            p = n

        return w0,w,wd,b,bd,z

    def sim(self,u):
        """ Simulate net given weights """
        w0,w,wd,b,bd,_ = self.extract(u)
        z = np.zeros((self.D,self.W,self.N))

        z[0] = self.sigma(w0.dot(self.x) + b[0])
        
        for i in range(self.D-1):
            z[i+1] = self.sigma(w[i].dot(z[i]) + b[i+1])

        y = self.tau(wd.dot(z[self.D-1])+bd)
        
        return y,z
    
    def eval_h(self,u):
        """ Evaluate constraints """
        W,D,N = self.W,self.D,self.N
        w0,w,wd,b,bd,z = self.extract(u)
        h = np.zeros((D-1,W,N))
        
        h1 = z[0]   - self.sigma(w0.dot(self.x)+b[0])
        
        for i in range(D-1):
            h[i] = z[i+1] - self.sigma(w[i].dot(z[i]) + b[i+1])

        return np.concatenate([h1.ravel(),h.ravel()])

    def eval_L(self,u,mu,l):
        """ Evaluate Lagrangian"""
        W,D,N = self.W,self.D,self.N
        w0,w,wd,b,bd,z = self.extract(u)
        
        F  = self.y - (wd.dot(z[D-1])+bd)
        F = F.ravel()/np.sqrt(mu)

        h = self.eval_h(u)
        h = h+l/mu

        return np.concatenate([h,F])

    
    def eval_J_L(self,u,mu,l):
        """ Evaluate Jacobian of Lagrangian """
        W,D,I,O,N,n_u = self.W,self.D,self.I,self.O,self.N,u.size
        w0,w,wd,b,bd,z = self.extract(u)

        # Init jacobian
        J = np.zeros((D*W*N+O*N,n_u))

        # Init horizontal slices of matrix

        h = []
        p,n = 0,0
        for i in range(D):
            n = p + W*N
            h.append(slice(p,n))
            p = n

        n = p + O*N
        F = slice(p,n)

        # W0
        w0_ = self.sigma_(w0.dot(self.x)+b[0])
        w0_ = -self.x*w0_[:,np.newaxis,:]
        w0_ = w0_*np.eye(W)[:,:,np.newaxis,np.newaxis]
        w0_ = np.swapaxes(w0_,1,2).reshape(I*W,W*N).transpose()

        p,n = 0,I*W
        J[h[0],p:n] = w0_
        p = n

        # Wi
        for i in range(D-1):
            w_ = self.sigma_(w[i].dot(z[i])+b[i+1])
            w_ = -z[i]*w_[:,np.newaxis,:]
            w_ = w_*np.eye(W)[:,:,np.newaxis,np.newaxis]
            w_ = np.swapaxes(w_,1,2).reshape(W*W,W*N).transpose()

            print(w_)

            n = p+W*W
            J[h[i+1],p:n] = w_
            p = n

        # WD
        wd_ = self.tau_(wd.dot(z[D-1])+bd)
        wd_ = -z[D-1]*wd_[:,np.newaxis,:]
        wd_ = wd_*np.eye(O)[:,:,np.newaxis,np.newaxis]
        wd_ = np.swapaxes(wd_,1,2).reshape(O*W,O*N).transpose()
        wd_ = wd_/np.sqrt(mu)

        n = p+O*W
        J[F,p:n] = wd_
        p = n

        #b0
        b0_ = -self.sigma_(w0.dot(self.x)+b[0])
        b0_ = np.eye(W)[:,np.newaxis,:]*b0_.transpose()
        b0_ = b0_.reshape(W*N,W)
                          
        n = p+W
        J[h[0],p:n] = b0_
        p = n


        #bi
        for i in range(D-1):
            b_ = -self.sigma_(w[i].dot(z[i])+b[i+1])
            b_ = np.eye(W)[:,np.newaxis,:]*b_.transpose()
            b_ = b_.reshape(W*N,W)

            n = p+W
            J[h[i+1],p:n] = b_
            p = n

        #bd
        bd_ = -self.tau_(wd.dot(z[D-1])+bd)
        bd_ = np.eye(O)[:,np.newaxis,:]*bd_.transpose()
        bd_ = bd_.reshape(O*N,O)
        bd_ = bd_/np.sqrt(mu)

        n = p+O
        J[F,p:n] = bd_
        p = n

        #z
        for i in range(D-1):
            z_ = -w[i].transpose()[:,:,np.newaxis]*self.sigma_(w[i].dot(z[i])+b[i+1])
            z_ = np.eye(N)*z_[:,:,:,np.newaxis]
            z_ = np.swapaxes(z_,1,2).reshape(W*N,W*N).transpose()

            n = p+W*N
            J[h[i],p:n] = np.eye(W*N)
            J[h[i+1],p:n] = z_
            p = n

        #zd
        zd_ = -wd.transpose()[:,:,np.newaxis]*self.tau_(wd.dot(z[D-1])+bd)
        zd_ = np.eye(N)*zd_[:,:,:,np.newaxis]
        zd_ = np.swapaxes(zd_,1,2).reshape(W*N,O*N).transpose()

        n = p+W*N
        J[h[D-1],p:n] = np.eye(W*N)
        J[F,p:n] = zd_/np.sqrt(mu)

        return J
    
  
def eval_L_U(u,mu,l,net):
    W,D,N = net.W,net.D,net.N
    w0,w,wd,b,bd,z = net.extract(u)

    F      = net.y  -          (UTPM.dot(wd,z[D-1]) + bd)
    h      = z[0]   - UTPM.tanh(UTPM.dot(w0,net.x)  + b[0])
    F,h = F.reshape((1,-1))/np.sqrt(mu),h.reshape((1,-1))
    for i in range(D-1):
        hi = z[i+1] - UTPM.tanh(UTPM.dot(w[i],z[i]) + b[i+1])
        hi = hi.reshape((1,-1))
        h = UTPM(np.concatenate([h.data,hi.data],3))

    h = h+l/mu
    return UTPM(np.concatenate([h.data,F.data],3))

