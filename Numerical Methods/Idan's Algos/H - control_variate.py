##################################################
# PRICING A DOWN-AND-OUT BARRIER PUT OPTION
# stock obeys GBM with r=0.1, s=0.4 (time unit = year = 252 days), current
# price 50. 60 day european put option, with strike 50, but a barrier at 30
# - below this the option gets knocked out thus reducing risk for seller.
# daily observation. use "control variable" of a regular european put


import numpy as np
from numpy import random as rn
import scipy.stats as ss

#Black and Scholes
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))

def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

def blsprice(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))

S0=50;
K=50;
r=0.1;
sigma=0.4;
T=60/252;       # up to here usual parameters for a put
B=35;           # barrier - see what happens when increase to 35 and 40!

N=60;           # number of observations
h=T/N;
V_P=blsprice('P',S0, K, r, sigma, T); #E[Y]
###################################################
# part 1: estimate the correlation of the barrier and vanilla options

M=5*10**3;
Z=rn.randn(M,N);
S=np.ones((M,N+1));

S[:,0]=S0*np.ones(M);
for i in range(0,N):
    S[:,i+1]=S[:,i]*np.exp((r-sigma**2/2)*h+sigma*np.sqrt(h)*Z[:,i]);

worst=np.min(S,1);
payoff=np.exp(-r*T)*(worst>B)*(S[:,N]<K)*(K-S[:,N]);#X
payoff2=np.exp(-r*T)*(S[:,N]<K)*(K-S[:,N]); #Y

q=np.cov(payoff,payoff2)
#print(q)# show the covariance matrix
c=-q[0,1]/q[1,1];
#print(c)             # show c

###################################################
# part 2: the real simulation

M=45*10**3;        # number of replications
Z=rn.randn(M,N);
S=np.ones((M,N+1));

S[:,0]=S0*np.ones(M);
for i in range(0,N):
    S[:,i+1]=S[:,i]*np.exp( (r-sigma**2/2)*h+sigma*np.sqrt(h)*Z[:,i]);

worst=np.min(S,1);
payoff=np.exp(-r*T)*(worst>B)*(S[:,N]<K)*(K-S[:,N]);#X
payoff2=np.exp(-r*T)*(S[:,N]<K)*(K-S[:,N]); #Y

corrected=payoff+c*(payoff2-V_P);#X_C=X+c*(Y-E[Y])

controlled=[np.mean(corrected),np.std(corrected)/np.sqrt(M)];controlled  # answers with control variable
uncontrolled=[np.mean(payoff),np.std(payoff)/np.sqrt(M)];uncontrolled       # answers without control variable
vanilla = [np.mean(payoff2),np.std(payoff2)/np.sqrt(M)];vanilla          # simulated answers for vanilla option
################################
print(controlled)
print(uncontrolled)
print(vanilla)