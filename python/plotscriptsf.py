import numpy as np
from matplotlib import pyplot

np.set_printoptions(linewidth=200)
with open('santafe600_2.npy','rb') as f:
    K = 4
    adamloss = [()]*K
    adamval  = [()]*K
    almhist  = [()]*K
    pred     = [()]*K
    for k in range(K):
        adamloss[k] = np.load(f)
        adamval[k]  = np.load(f)
        almhist[k]  = np.load(f)
        pred[k]     = np.load(f)
    #t = np.load(f)

lasertrain = np.genfromtxt('lasertrain.csv')
lasertest = np.genfromtxt('laserpred.csv')
lm,ls = np.mean(lasertrain),np.std(lasertrain)
Ntst = lasertest.size

#pyplot.semilogy(np.arange(adamloss[2].size),adamloss[2],np.arange(adamval[2].size)*200,adamval[2])
#pyplot.title('ADAM convergence')
#pyplot.xlabel('epochs')
#pyplot.gcf().subplots_adjust(bottom=0.15)
#
#pyplot.figure()
#alm_epochs = almhist[0].shape[0]
#pyplot.semilogy(np.arange(alm_epochs),almhist[0][:,0],np.arange(alm_epochs),almhist[0][:,3])
#pyplot.title('ALM convergence')
#pyplot.xlabel('epochs')
#pyplot.gcf().subplots_adjust(bottom=0.15)

pyplot.plot(np.arange(Ntst),lasertest,'k-')
pyplot.plot(np.arange(Ntst),pred[3][:Ntst]*ls+lm,'r--')
pyplot.title('ADAM prediction')
pyplot.figure()
pyplot.plot(np.arange(Ntst),lasertest,'k-')
pyplot.plot(np.arange(Ntst),pred[2][Ntst:]*ls+lm,'r--')
pyplot.title('ALM prediction')
pyplot.show()

res = np.empty((2,6))
res[0,0] = np.mean([loss[-1] for loss in adamloss])
res[0,1] = np.min( [loss[-1] for loss in adamloss])
res[0,2] = np.mean(t[:,4])
res[0,3] = np.min(t[:,4])
res[0,4] = np.mean(t[:,2])
res[0,5] = np.mean(t[:,0])

res[1,0] = np.mean([hist[-1,0] for hist in almhist])
res[1,1] = np.min( [hist[-1,0] for hist in almhist])
res[1,2] = np.mean(t[:,5])
res[1,3] = np.min(t[:,5])
res[1,4] = np.mean(t[:,3])
res[1,5] = np.mean(t[:,1])

np.set_printoptions(linewidth=200,precision=4)
print(res)

evalj = np.concatenate([hist[:,2] for hist in almhist]).reshape((5,-1),order='F')
print(np.mean(evalj,axis=1))




#    t.append(np.load(f))

#    res.append(np.zeros((K,4)))
#    for k in range(K):
#        res[i][k,:] = np.r_[hist[i][k][0][-1],hist[i][k][1][-1,:]]
#        res[i][k,3] = int(np.sum(hist[i][k][1][:,2]))
            
#
#np.set_printoptions(precision=2)
#conv0 = res[0][:,0]<.25
#conv1 = res[0][:,1]<.25
#p = np.array([np.mean(res[0][conv0,0]), min(res[0][:,0]), np.mean(t[0][conv0,0]), np.mean(t[0][conv0,2]),np.count_nonzero(conv0),
#              np.mean(res[0][conv1,1]), min(res[0][:,1]), np.mean(t[0][conv1,1]), np.mean(t[0][conv1,3]),np.count_nonzero(conv1)])
#print(p.reshape((2,-1)))
#conv0 = res[1][:,0]<.25
#conv1 = res[1][:,1]<.25
#p = np.array([np.mean(res[1][conv0,0]), min(res[1][:,0]), np.mean(t[1][conv0,0]), np.mean(t[1][conv0,2]),np.count_nonzero(conv0),
#              np.mean(res[1][conv1,1]), min(res[1][:,1]), np.mean(t[1][conv1,1]), np.mean(t[1][conv1,3]),np.count_nonzero(conv1)])
#print(p.reshape((2,-1)))
#conv0 = res[2][:,0]<.25
#conv1 = res[2][:,1]<.25
#p = np.array([np.mean(res[2][conv0,0]), min(res[2][:,0]), np.mean(t[2][conv0,0]), np.mean(t[2][conv0,2]),np.count_nonzero(conv0),
#              np.mean(res[2][conv1,1]), min(res[2][:,1]), np.mean(t[2][conv1,1]), np.mean(t[2][conv1,3]),np.count_nonzero(conv1)])
#print(p.reshape((2,-1)))
#
#print(res[0].reshape((-1,2)))
#print(np.mean(res[1]+res[4],0)/2)
#print(np.mean(res[2]+res[5],0)/2)
#print(np.c_[res[1][:,0],res[4][:,0]])
#print(np.c_[res[2][:,0],res[5][:,0]])

#pyplot.hist(res[2][:,1], bins = 10)
#pyplot.xlabel("MSE")
#pyplot.ylabel("# runs")
#pyplot.tight_layout()
#pyplot.gcf().subplots_adjust(bottom=0.15)
#pyplot.show()
    



            

