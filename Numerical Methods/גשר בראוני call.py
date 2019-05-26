import numpy as np
from numpy import random as rn
from scipy import stats as ss


S0=50;
K=50;
r=0.1;
sig=0.4;
T=60/252;
N=60;
M=10**4;
B=40
h=T/N;


#probability that the option has value (K is under ST )
z0=(np.log(K/S0)-(r-sig**2/2)*T)/(sig*np.sqrt(T)) ;
p =ss.norm.cdf(z0) ;


#sampling Z | Z>z0 using the inverse transform method
U = rn.rand(M);
ZT = ss.norm.ppf( p+U*(1-p));

#building a trajectory (maslul) of the Brownian bridge
W=np.zeros((M,N+1));
S=np.ones((M,N+1));

S[:,0]=S0*np.ones(M);
W[:,N]=np.sqrt(T)*ZT;
for i in range(1,N):
	# build column i of Z − using previous column and last column
	t = (i-1)*T/N;
	a = (i-2)*T/N;
	b = T;
	sigma = np.sqrt( (t-a)*(b-t)/(b-a) );
	mu = W[:,i-1] + (t-a)/(b-a)*(W[:,N]-W[:,i-1]) ;
	W[:,i] = mu + sigma*rn.randn(M);
	S[:,i]=S[:,i-1]*np.exp((r-sigma**2/2)*h+sig*(W[:,i]-W[:,i-1]));

S[:,N]=S[:,N-1]*np.exp((r-sigma**2/2)*h+sig*(W[:,N]-W[:,N-1]));
worst=np.min(S,1);
payoff=np.exp(-r*T)*(worst>B)*(S[:,N]>K)*(S[:,N]-K);#(S[:,N]>K) is not required
print([(1-p)*np.mean(payoff),np.std((1-p)*payoff)/np.sqrt(M)]);
