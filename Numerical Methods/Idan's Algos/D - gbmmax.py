#------------------------------------------------
# Pricing a Max Option on an asset that obeys GBM
#------------------------------------------------

# using both EM and the exact GBM formula to get prices
# (the error from taking the "max" over a discrete set
#  of points is more significant than the EM error)
import numpy as np
from numpy import random as rn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = 0.02;
mu = 0.02;
sigma = 0.2;
T = 2;
S0 = 1.5;
K = 1.2;

M = 10**6;     # keep large as want to see the deterministic error
N = 5;          # vary to see the deterministic error
h = T/N;
t = h*np.arange(N+1);

S = S0*np.ones((M,N+1));
S2 = S0*np.ones((M,N+1));
dW = np.sqrt(h)*rn.randn(M,N);

# Euler Maruyama (EM)
for i in range(0,N):
   S[:,i+1] = S[:,i]*(1 + h * r  + sigma * dW[:,i]);

# GBM
for i in range(0,N):
   S2[:,i+1] = S[:,i]*np.exp((r-(sigma**2)/2)*h + sigma*dW[:,i]);

j=15
fig=plt.figure()
ax = fig.add_subplot(111)
ax.plot(t,S[i,:])
ax.plot(t,S2[i,:]);
ax.legend(['EM','GBM'])
fig.savefig('GBMMAX.png')

##
MM = np.max(S,1)-K;
NN = (MM > 0)*MM*np.exp(-mu*T) ;
val1=[np.mean(NN), np.std(NN)/np.sqrt(M)];

print(val1)

MM2 = np.max(S2,1)-K;
NN2 = (MM2 > 0)*MM2*np.exp(-mu*T) ;
val2=[np.mean(NN2), np.std(NN2)/np.sqrt(M)];

print(val2)