"""
תשס"ט מועד ב שאלה 2
"""
import numpy as np
from numpy import random as rn

def s_sigma(s0,sigma0,T,r,b,a,M,n):
    h=T/n
    S=s0*np.ones((M,n+1))
    sigma=sigma0*np.ones((M,n+1))
    z1=rn.randn(M,n)
    z2=rn.randn(M,n)
    for i in range(0,n):
        S[:,i+1]=S[:,i]*np.exp((r-sigma[:,i]**2/2)*h+sigma[:,i]*np.sqrt(h)*z1[:,i])
        sigma[:,i+1]=sigma[:,i]+(b-sigma[:,i])*h+a*sigma[:,i]*np.sqrt(h)*z2[:,i]
    p=np.exp(-r*T)*S[:,-1]
    return("E[S(T)]=", np.mean(p),np.std(p)/np.sqrt(M)) 

print(s_sigma(1,0.1,300,0.2,0.01,0.02,50000,1000))
