import numpy as np

import matplotlib #redundant with Jupyter
#matplotlib.use("Agg")
import matplotlib.pyplot as plt #will not be in the exam



import numpy as np
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


S=112.6;#price
K=100;#strike
r=0.02;#rate
sigma=np.linspace(0.01,1,101);
T=2/12;#time

C=blsprice('C',S,K,r,sigma,T)
P=blsprice('P',S,K,r,sigma,T)
##
#[Call, Put] = blsprice(Price, Strike, Rate, Time, Volatility, Yield)


#specific for pythonanywhere - not for the exam
fig=plt.figure()
ax = fig.add_subplot(111)
ax.plot(sigma,C)
fig.savefig('Call.png')
#plt.show()
ax.cla()
ax.plot(sigma,P);
fig.savefig('Put.png')
#plt.show()
