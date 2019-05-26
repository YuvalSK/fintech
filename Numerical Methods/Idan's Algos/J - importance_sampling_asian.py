import numpy as np
from numpy import random as rn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S0 = 1;
K = 1.3;
sigma = 0.1;
r = 0.02;
T = 2;
N = 24;
mu = np.linspace(0,20,400)/N ;
h = T/N;


val = np.zeros(np.shape(mu));
err= np.zeros(np.shape(mu));

M = 5*10**4;
Z = rn.randn(M,N);
S_star = S0 * np.ones((M,N+1));

for i in range(0,np.size(mu)):
    newZ=mu[i]+Z
    for j in range(0,24):
       S_star[:,j+1] = S_star[:,j] * np.exp( (r-sigma**2/2)*h + sigma*np.sqrt(h)*newZ[:,j]);
    Sbar = np.mean(S_star,1);
    C = np.exp(-r*T+N*mu[i]**2/2)*(Sbar-K)*(Sbar>K)*np.exp(-mu[i]*np.sum(newZ,1));
    val[i]=100*np.mean(C);
    err[i]=100*np.std(C)/np.sqrt(M);

fig=plt.figure()
ax = fig.add_subplot(111)
ax.plot(mu,val)
fig.suptitle('Asian Option price as function of mu')
fig.savefig('ASIAN_Improtance_Sampling1.png')


ax.cla()
ax.plot(mu,err)
fig.suptitle('Asian error as function of mu')
fig.savefig('ASIAN_Improtance_Sampling2.png')