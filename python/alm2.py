import numpy as np
import scipy.linalg as la
import scipy.optimize as op
from algopy import UTPM
import matplotlib.pyplot as pyplot


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

        return np.concatenate([F,h])

    
    def eval_J_L(self,u,mu,l):
        """ Evaluate Jacobian of Lagrangian """
        W,D,I,O,N,n_u = self.W,self.D,self.I,self.O,self.N,u.size
        w0,w,wd,b,bd,z = self.extract(u)

        # Init jacobian
        J = np.zeros((D*W*N+O*N,n_u))

        # Init horizontal slices of matrix
        p,n = 0,O*N
        F = slice(p,n)
        p = n

        h = []
        for i in range(D):
            n = p + W*N
            h.append(slice(p,n))
            p = n

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
    return UTPM(np.concatenate([F.data,h.data],3))

def solve_ls(u,fun,jac,tol):
    J = jac(u)
    m,n = np.shape(J)
    lm,nu = 0.01,2
    while(True):
        lm = lm/nu
        J = jac(u)
        r = fun(u)
        print(la.norm(J))
        if(la.norm(J)<tol):
            break

        b = la.solve(np.dot(J.transpose(),J)+lm*np.eye(n),np.dot(J.transpose(),r))

        while(la.norm(fun(u-b))>la.norm(fun(u))):
            lm = lm*nu
            b = la.solve(np.dot(J.transpose(),J)+lm*np.eye(n),np.dot(J.transpose(),r))
            print(la.norm(r))
        print()
        u = u-b


#        lm = lm/nu
#        J = jac(u)
#        r = fun(u)
#        print(la.norm(J),la.norm(r),np.linalg.norm(J.transpose()))
#        if(la.norm(J)<tol):
#            break
#
#        Jl = np.concatenate((J,lm*np.eye(n)))
#        Q,R = la.qr(Jl,mode='economic')
#        rl = np.dot(Q.transpose(),np.concatenate((r,np.zeros((n,)))))
#        b = la.solve_triangular(R,rl)
#        u0 = u-b
#
#        while(la.norm(fun(u0))>la.norm(fun(u))):
#            lm = lm*nu
#            Jl = np.concatenate((J,lm*np.eye(n)))
#            Q,R = la.qr(Jl,mode='economic')
#            rl = np.dot(Q.transpose(),np.concatenate((r,np.zeros((n,)))))
#            b = la.solve_triangular(R,rl)
#            u0 = u-b
#        u = u0
#        print()

    return u




I,O,W,D,N, = 2,2,3,2,7
mu = 10

shape = (W,D,I,O)
sigma  = lambda x: np.tanh(x)
sigma_ = lambda x: 2/(np.cosh(2*x)+1)

tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)

x = np.random.random_sample((I,N))
#x = np.linspace(0,1,N).reshape((1,N)).repeat(2,0)
y = -np.sin(.8*np.pi*x)
y[1,:] = -y[1,:]
u = np.random.random_sample((W*W*(D-1) + D*W + I*W + O*W + O + D*W*N,))
l = np.random.random_sample((D*W*N,))

net = Net(shape,sigma,sigma_,tau,tau_,x,y)

_,z = net.sim(u)
n_u = u.size
u[n_u-W*D*N:n_u] = z.ravel()

L = net.eval_L(u,mu,l)
J = net.eval_J_L(u,mu,l)

u_U = UTPM.init_jacobian(u)
L_U = eval_L_U(u_U,mu,l,net)
J_U = UTPM.extract_jacobian(L_U)
J_U = J_U.reshape(J_U.shape[1:3])

print(np.allclose(L,np.ravel(L_U.data[0,0])))
print(np.allclose(J,J_U))

Jnz = J != 0

T = Jnz[0:O*N,:]
Jnz = Jnz[O*N+1:-1,:]
Jnz = np.concatenate((Jnz,T))
pyplot.matshow(Jnz,cmap="binary")
pyplot.show()

np.set_printoptions(linewidth=200,precision=4)
print(J.shape)
Jm,Jn = J.shape
for i in range(Jm):
    for j in range(Jn):
        if not np.isclose(J[i,j],J_U[i,j]):
            print("X",end='')
        elif J[i,j]:
            print('#',end='')
        else:
            print(' ',end='')
        
    print()

#mu = 10
#eta,omega = np.power(1/mu,0.1),1/mu
#eta_ = 1e-4
#for k in range(10):
#    fun = lambda u: net.eval_L(u,mu,l)
#    jac = lambda u: net.eval_J_L(u,mu,l)
#    print(jac(u).shape)
#    sol = op.least_squares(fun,u,jac,ftol=None,xtol=None,gtol=1/mu)
#    u = sol.x
#    #u = solve_ls(u,fun,jac,1/mu)
#
#    h = net.eval_h(u)
#    print(la.norm(h))
#    if la.norm(h) <= eta:
#        if la.norm(h) <= eta_:
#            break
#        l = l - mu*h
#        eta,omega = 1/mu,1/mu/mu
#    else:
#        mu = 100*mu
#        eta,omega = np.power(1/mu,0.1),1/mu
#        
#y,_ = net.sim(u)
#
#pyplot.plot(x.ravel(),net.y.ravel(),'k+',x.ravel(),y.ravel(),'r-')
#pyplot.show()

