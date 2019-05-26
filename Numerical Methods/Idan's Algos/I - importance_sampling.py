# price a deep out of the money call option using importance sampling
# look at E[ exp(-rT) (S(T)-K)_+ exp(mu^2/2 - mu Z) ]
# where Z ~ N( mu, 1)
# below we write "mu + Z" instead of Z ... and plot the answer and the
# error for different values of mu
import numpy as np
from numpy import random as rn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S0 = 1;
K = 1.4;
sigma = 0.1;
r = 0.02;
T = 2;
mu = np.linspace(0,5,250);
val = np.zeros(np.shape(mu));
err= np.zeros(np.shape(mu));


M = 5*10**4;
Z = rn.randn(M,1);

for i in range(0,np.size(mu)):
	newZ=mu[i]+Z #~N(mu,1) - change of r.v.
	ST_star = S0 * np.exp( (r-sigma**2/2)*T + sigma*np.sqrt(T)*newZ);
	C = np.exp(-r*T+mu[i]**2/2)*(ST_star-K)*(ST_star>K)*np.exp(-mu[i]*newZ) ;
	val[i]=100*np.mean(C);
	err[i]=100*np.std(C)/np.sqrt(M);



fig=plt.figure()
ax = fig.add_subplot(111)
ax.plot(mu,val)
fig.suptitle('Option price as function of mu')
fig.savefig('Improtance_Sampling1.png')


ax.cla()
ax.plot(mu,err)
fig.suptitle('error as function of mu')
fig.savefig('Improtance_Sampling2.png')
