# Vasicek process
# dX = a(c - X) dt + b dW
# compute the probability of the process going over a certain level
import numpy as np
from numpy import random as rn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
a = 1 ;
b = 4 ;
c = 8 ;

T = 5 ;
N = 500 ;
h = T/N ;
t = h*np.arange(N+1);

M = 10**4 ;
X0 = 1 ;

X = X0*np.ones((M,N+1));
X2 = X0*np.ones((M,N+1));

Z = rn.randn(M,N);
Y = np.zeros((M,N+1));
dW=np.sqrt(h)*Z;
for i in range(0,N):
    X[:,i+1] = X[:,i] + a*(c-X[:,i])*h + b *dW[:,i] ; #EM - we can do beter
    #simulate with the half analytic solution
    Y[:,i+1] = Y[:,i] + np.exp(a*i*h)*dW[:,i];
    X2[:,i+1] = c +(X0-c)*np.exp(-a*(i+1)*h)+b*np.exp(-a*(i+1)*h)*Y[:,i+1];

##
j=15
fig=plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,X[i,:])
ax.plot(t,X2[i,:]);
ax.legend(['EM','half analytic'])
fig.savefig('VASICEK.png')

##
p = np.sum(np.max(X,1)>9)/M ;
print(p)
err = np.sqrt(p*(1-p))/np.sqrt(M);
print(err)