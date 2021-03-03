import numpy as np
import net
import scipy.linalg as la
import scipy.optimize as op
from algopy import UTPM
import matplotlib.pyplot as pyplot


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




I,O,W,D,N, = 2,2,4,3,11
mu = 10

shape = (W,D,I,O)
sigma  = lambda x: np.tanh(x)
sigma_ = lambda x: 2/(np.cosh(2*x)+1)

tau  = lambda x: x
tau_ = lambda x: np.ones(x.shape)

x = np.linspace(0,1,N).reshape((1,N)).repeat(2,0)
y = -np.sin(.8*np.pi*x)
y[1,:] = -y[1,:]
u = np.random.random_sample((W*W*(D-1) + D*W + I*W + O*W + O + D*W*N,))
l = np.random.random_sample((D*W*N,))

nn = net.Net(shape,sigma,sigma_,tau,tau_,x,y)

_,z = nn.sim(u)
n_u = u.size
u[n_u-W*D*N:n_u] = z.ravel()

L = nn.eval_L(u,mu,l)
J = nn.eval_J_L(u,mu,l)

u_U = UTPM.init_jacobian(u)
L_U = net.eval_L_U(u_U,mu,l,nn)
J_U = UTPM.extract_jacobian(L_U)
J_U = J_U.reshape(J_U.shape[1:3])

print(np.allclose(L,np.ravel(L_U.data[0,0])))
print(np.allclose(J,J_U))

mu = 10
eta,omega = np.power(1/mu,0.1),1/mu
eta_ = 1e-4
for k in range(10):
    fun = lambda u: nn.eval_L(u,mu,l)
    jac = lambda u: nn.eval_J_L(u,mu,l)
    print(jac(u).shape)
    sol = op.least_squares(fun,u,jac,ftol=None,xtol=None,gtol=1/mu)
    u = sol.x
    #u = solve_ls(u,fun,jac,1/mu)

    h = nn.eval_h(u)
    print(la.norm(h))
    if la.norm(h) <= eta:
        if la.norm(h) <= eta_:
            break
        l = l - mu*h
        eta,omega = 1/mu,1/mu/mu
    else:
        mu = 100*mu
        eta,omega = np.power(1/mu,0.1),1/mu
        
y,_ = nn.sim(u)

pyplot.plot(x[0,:].ravel(),nn.y[0,:].ravel(),'k+',x[0,:].ravel(),y[0,:].ravel(),'r-')
pyplot.plot(x[1,:].ravel(),nn.y[1,:].ravel(),'k+',x[1,:].ravel(),y[1,:].ravel(),'r-')
pyplot.show()

