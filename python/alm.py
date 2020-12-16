import numpy as np
import scipy.linalg as la
import scipy.optimize as op
from algopy import UTPM
import matplotlib.pyplot as pyplot


class Net:

    def __init__(self,shape, sigma, sigma_, x, y):
        self.sigma  = sigma
        self.sigma_ = sigma_
        self.shape = shape
        self.x = x
        self.y = y
        self.W  = shape[0]
        self.D  = shape[1]
        self.N  = x.size

    def extract(self,u):
        W,D,N = self.W,self.D,self.N

        p,n = 0,W
        w1 = u[p:n].reshape((W,1))
        p = n
    
        w = []
        for i in range(D-1):
            n = p + W*W
            w.append(u[p:n].reshape((W,W)))
            p = n

        n = p + W
        wn =  u[p:n].reshape((1,W))
        p = n
        
        b = []
        for i in range(D):
            n = p + W
            b.append(u[p:n].reshape((W,1)))
            p = n

        n = p + 1
        bn = u[p:n].reshape((1,1))
        p = n
        
        z = []
        for i in range(D):
            n = p + W*N
            z.append(u[p:n].reshape((W,N)))
            p = n

        return w1,w,wn,b,bn,z

    def sim(self,u):
        w1,w,wn,b,bn,_ = self.extract(u)
        z = np.zeros((self.D,self.W,self.N))

        z[0] = self.sigma(w1.dot(self.x) + b[0])
        
        for i in range(self.D-1):
            z[i+1] = self.sigma(w[i].dot(z[i]) + b[i+1])
            y = wn.dot(z[self.D-1])+bn
        
        return y,z
    
    def eval_h(self,u):
        W,D,N = self.W,self.D,self.N
        w1,w,wn,b,bn,z = self.extract(u)
        h = np.zeros((D-1,)+z[0].shape)
        
        h1 = z[0]   - self.sigma(w1.dot(self.x)+b[0])
        
        for i in range(D-1):
            h[i] = z[i+1] - self.sigma(w[i].dot(z[i]) + b[i+1])

        return np.concatenate([h1.ravel(),h.ravel()])

    def eval_L(self,u,mu,l):
        W,D,N = self.W,self.D,self.N
        w1,w,wn,b,bn,z = self.extract(u)
        
        F  = self.y - (wn.dot(z[D-1])+bn)
        F = F.ravel()/np.sqrt(mu)

        h = self.eval_h(u)
        h = h+l/mu

        return np.concatenate([F,h])

    
    def eval_J_L(self,u,mu,l):
        W,D,N,n_u = self.W,self.D,self.N,u.size
        w1,w,wn,b,bn,z = self.extract(u)
        J = np.zeros((D*W*N+N,n_u))
        p,n = 0,N
        F = slice(p,N)
        p = n

        h = []
        for i in range(D):
            n = p + W*N
            h.append(slice(p,n))
            p = n

        w1_ = -self.x*self.sigma_(w1.dot(self.x)+b[0])
        w1_ = np.eye(W)[:,np.newaxis,:]*w1_.transpose()
        w1_ = w1_.reshape(W*N,W)

        p,n = 0,W
        J[h[0],p:n] = w1_
        p = n

        for i in range(D-1):
            w_ = self.sigma_(w[i].dot(z[i])+b[i+1])
            w_ = -z[i]*w_[:,np.newaxis,:]
            w_ = w_*np.eye(W)[:,:,np.newaxis,np.newaxis]
            w_ = np.swapaxes(w_,1,2).reshape(W*W,W*N).transpose()

            n = p+W*W
            J[h[i+1],p:n] = w_
            p = n

        wn_ = -z[D-1].transpose()/np.sqrt(mu)

        n = p+W
        J[F,p:n] = wn_
        p = n

        b1_ = -self.sigma_(w1.dot(x)+b[0])
        b1_ = np.eye(W)[:,np.newaxis,:]*b1_.transpose()
        b1_ = b1_.reshape(W*N,W)
                          
        n = p+W
        J[h[0],p:n] = b1_
        p = n



        for i in range(D-1):
            b_ = -self.sigma_(w[i].dot(z[i])+b[i+1])
            b_ = np.eye(W)[:,np.newaxis,:]*b_.transpose()
            b_ = b_.reshape(W*N,W)

            n = p+W
            J[h[i+1],p:n] = b_
            p = n

        bn_ = -np.ones((N,1))/np.sqrt(mu)

        n = p+1
        J[F,p:n] = bn_
        p = n

        for i in range(D-1):
            z_ = -w[i].transpose()[:,:,np.newaxis]*self.sigma_(w[i].dot(z[i])+b[i+1])
            z_ = np.eye(N)*z_[:,:,:,np.newaxis]
            z_ = np.swapaxes(z_,1,2).reshape(W*N,W*N).transpose()

            n = p+W*N
            J[h[i],p:n] = np.eye(W*N)
            J[h[i+1],p:n] = z_
            p = n

        zn_ = -np.eye(N)*wn[:,:,np.newaxis,np.newaxis]/np.sqrt(mu)
        zn_ = np.swapaxes(zn_,1,2).reshape(N,W*N)

        n = p+W*N
        J[h[D-1],p:n] = np.eye(W*N)
        J[F,p:n] = zn_

        return J
    
  
def eval_L_U(u,mu,l,net):
    W,D,N = net.W,net.D,net.N
    w1,w,wn,b,bn,z = net.extract(u)

    F      = net.y  -          (UTPM.dot(wn,z[D-1]) + bn)
    h      = z[0]   - UTPM.tanh(UTPM.dot(w1,net.x)  + b[0])
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




W,D,N, = 3,2,33
mu = 10
shape = (W,D)
sigma = lambda x: np.tanh(x)
sigma_ = lambda x: 2/(np.cosh(2*x)+1)

x = np.reshape(np.linspace(0,1,N),[1,N])
y = -np.sin(.8*np.pi*x)
u = np.random.random_sample((W+(D-1)*W*W+W+D*W+1+D*W*N,))
l = np.random.random_sample((D*W*N,))

net = Net(shape,sigma,sigma_,x,y)

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

print(J.shape)
#Jm,Jn = J.shape
#for i in range(Jm):
#    for j in range(Jn):
#        if J[i,j]:
#            print('#',end='')
#        else:
#            print(' ',end='')
#    print()

mu = 10
eta,omega = np.power(1/mu,0.1),1/mu
eta_ = 1e-4
for k in range(10):
    fun = lambda u: net.eval_L(u,mu,l)
    jac = lambda u: net.eval_J_L(u,mu,l)
    print(jac(u).shape)
    #sol = op.least_squares(fun,u,jac,ftol=None,xtol=None,gtol=1/mu)
    #u = sol.x
    u = solve_ls(u,fun,jac,1/mu)

    h = net.eval_h(u)
    print(la.norm(h))
    if la.norm(h) <= eta:
        if la.norm(h) <= eta_:
            break
        l = l - mu*h
        eta,omega = 1/mu,1/mu/mu
    else:
        mu = 100*mu
        eta,omega = np.power(1/mu,0.1),1/mu
        
y,_ = net.sim(u)

pyplot.plot(x.ravel(),net.y.ravel(),'k+',x.ravel(),y.ravel(),'r-')
pyplot.show()

