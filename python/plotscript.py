import numpy as np
from matplotlib import pyplot

np.set_printoptions(linewidth=200)
with open('testrelun.npy','rb') as f:
    hist = []
    t = []
    res = []
    K = 20
    for i in range(3):
        hist.append([])   
        for k in range(K):
            h1 = np.load(f)
            h2 = np.load(f)
            hist[i].append((h1,h2))

        t.append(np.load(f))

        res.append(np.zeros((K,4)))
        for k in range(K):
            res[i][k,:] = np.r_[hist[i][k][0][-1],hist[i][k][1][-1,:]]
            res[i][k,3] = int(np.sum(hist[i][k][1][:,2]))
            

np.set_printoptions(precision=2)
conv0 = res[0][:,0]<.25
conv1 = res[0][:,1]<.25
p = np.array([np.mean(res[0][conv0,0]), min(res[0][:,0]), np.mean(t[0][conv0,0]), np.mean(t[0][conv0,2]),np.count_nonzero(conv0),
              np.mean(res[0][conv1,1]), min(res[0][:,1]), np.mean(t[0][conv1,1]), np.mean(t[0][conv1,3]),np.count_nonzero(conv1)])
print(p.reshape((2,-1)))
conv0 = res[1][:,0]<.25
conv1 = res[1][:,1]<.25
p = np.array([np.mean(res[1][conv0,0]), min(res[1][:,0]), np.mean(t[1][conv0,0]), np.mean(t[1][conv0,2]),np.count_nonzero(conv0),
              np.mean(res[1][conv1,1]), min(res[1][:,1]), np.mean(t[1][conv1,1]), np.mean(t[1][conv1,3]),np.count_nonzero(conv1)])
print(p.reshape((2,-1)))
conv0 = res[2][:,0]<.25
conv1 = res[2][:,1]<.25
p = np.array([np.mean(res[2][conv0,0]), min(res[2][:,0]), np.mean(t[2][conv0,0]), np.mean(t[2][conv0,2]),np.count_nonzero(conv0),
              np.mean(res[2][conv1,1]), min(res[2][:,1]), np.mean(t[2][conv1,1]), np.mean(t[2][conv1,3]),np.count_nonzero(conv1)])
print(p.reshape((2,-1)))

#print(res[0].reshape((-1,2)))
#print(np.mean(res[1]+res[4],0)/2)
#print(np.mean(res[2]+res[5],0)/2)
#print(np.c_[res[1][:,0],res[4][:,0]])
#print(np.c_[res[2][:,0],res[5][:,0]])

pyplot.hist(res[2][:,1], bins = 10)
pyplot.xlabel("MSE")
pyplot.ylabel("# runs")
pyplot.tight_layout()
pyplot.gcf().subplots_adjust(bottom=0.15)
pyplot.show()
    



            

