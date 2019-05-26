# price a call option on the average price of two shares
# plot result as a function of the correlation coefficient
import numpy as np
from numpy import random as rn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S10 = 15;
S20 = 30;
r = 0.05;
sigma1 = 0.1 ;
sigma2 = 0.08;
rho = np.linspace(-1,1,50);
T = 1;
K = 24;

M = 10**4;


g4 = np.zeros(np.shape(rho));
g5 = np.zeros(np.shape(rho));
Z = rn.randn(M,2);

for i in range(0,len(rho)):
   dW1=np.sqrt(T) * Z[:,0];
   V2 = rho[i] * Z[:,0] +  np.sqrt(1-rho[i]**2) * Z[:,1] ; #by Cholesky's decomposition
   dW2=np.sqrt(T) * V2; #WT-W0

   S1 = S10 * np.exp( (r-sigma1**2/2)*T + sigma1 * dW1 );
   S2 = S20 * np.exp( (r-sigma2**2/2)*T + sigma2 * dW2 );
   A = (S1+S2)/2 ;
   P = np.exp(-r*T)*(A-K)*(A>K);
   g4[i] = np.mean(P);
   g5[i] = np.std(P)/np.sqrt(M);

fig=plt.figure()
ax = fig.add_subplot(111)
ax.plot(rho,g4)
fig.suptitle('Option price as function of correlation')
fig.savefig('MultiD1.png')

print ('price: {0}, Error: {1}'.format(np.mean(g4), np.mean(g5)))
ax.cla()
ax.plot(rho,g5)
fig.suptitle('Option price as function of correlation')
fig.savefig('MultiD2.png')