"""
מבחן תשס"ט מועד א שאלה 2
"""
import numpy as np
from numpy import random as rn

def call_L_option(s0,r,sigma,T,L,K):

    M=50000
    n=T
    S=s0*np.ones((M,n+1))
    dw=rn.randn(M,n)
    h=1
    for i in range(0,n):
        S[:,i+1]=S[:,i]*np.exp((r-sigma**2/2)*h+sigma*np.sqrt(h)*dw[:,i])
    mm=np.mean(S[:,T-5:],1)-L  
    nn=(mm>0)*(S[:,-1]-K)*(S[:,-1]>K)*np.exp(-r*T)
    V=[np.mean(nn),np.std(nn)/np.sqrt(M)]
    return("V=", V)

print(call_L_option(1,0.2,0.1,365,1,1.1))
