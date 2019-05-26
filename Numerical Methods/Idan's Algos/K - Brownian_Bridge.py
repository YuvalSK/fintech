import numpy as np
from numpy import random as rn
from scipy import stats as ss
import matplotlib.pyplot as plt
S0=50;
K=50;
r=0.1;
sig=0.4;
T=60/252;
N=60;
M=10**4;

#probability that the option has value (K is above ST )
z0=(np.log(K/S0)-(r-sig**2/2)*T)/(sig*np.sqrt(T)) ;
p = ss.norm.cdf(z0) ;

#sampling Z | Z<z0 using the inverse transform method
U = rn.rand(M);
ZT = ss.norm.ppf( U * p);

#building a trajectory (maslul) of the Brownian bridge
W=np.zeros((M,N+1));
W[:,N]=np.sqrt(T)*ZT;

for i in range(1,N):
	# build column i of Z − using previous column and last column
	t = (i-1)*T/N;
	tm = (i-2)*T/N;
	tp = T;
	sigma = np.sqrt( (t-tm)*(tp-t)/(tp-tm) );
	mu = W[:,i-1] + (t-tm)/(tp-tm)*(W[:,N]-W[:,i-1]) ;
	W[:,i] = mu + sigma*rn.randn(M);

