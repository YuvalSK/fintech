"""
מבחן תש"ע מועד ב' שאלה 2א
"""
import numpy as np
from numpy import random as rn
import scipy.stats as ss

def GBM_option(s0,r,sigma,K):
    T=252
    M=10000
    n=2*T
    S=s0*np.ones((M,n+1))
    V=np.ones((M,1))
    dw=rn.randn(M,n)
    h=1
    for i in range(0,n):
        S[:,i+1]=S[:,i]*np.exp((r-sigma**2/2)*h+sigma*np.sqrt(h)*dw[:,i])
    for j in range(0,M):
        A=np.mean(S[j,:T])-K
        if A>0:
            V[j]=A*np.exp(-r*T)
        else:
            V[j]=(S[j,-1]-K)*(S[j,-1]>K)*np.exp(-r*2*T)
    V=[np.mean(V),np.std(V)/np.sqrt(M)]
    return("V=", V)

print(GBM_option(1,0.0002,0.001,1.05))

"""
מבחן תש"ע מועד ב' שאלה 2ב
conditional monte carlo
"""
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))

def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
def blsprice(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r*T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))

def CMC_GBM_option(s0,r,sigma,K):
    T=252
    M=10000
    n=T
    S=s0*np.ones((M,n+1))
    V=np.ones((M,1))
    dw=rn.randn(M,n)
    h=1
    for i in range(0,n):
        S[:,i+1]=S[:,i]*np.exp((r-sigma**2/2)*h+sigma*np.sqrt(h)*dw[:,i])
    for j in range(0,M):
        A=np.mean(S[j,:T])-K
        if A>0:
            V[j]=A*np.exp(-r*T)
        else:
            V[j]=blsprice("C",S[j,T], 1.05, 0.0002, 0.001, T)*np.exp(-r*T)
    V=[np.mean(V),np.std(V)/np.sqrt(M)]
    return("V=", V)
print(CMC_GBM_option(1,0.0002,0.001,1.05))
