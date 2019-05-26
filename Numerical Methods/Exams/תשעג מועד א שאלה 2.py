"""
תשע"ג מועד א' שאלה 2
"""
import numpy as np
from numpy import random as rn

import scipy.stats as ss

T=1
r=0.1
sigma11=0.3
sigma12=-0.4
sigma21=-0.1
sigma22=0.4
sigma1=np.sqrt(sigma12**2+sigma11**2)
sigma2=np.sqrt(sigma21**2+sigma22**2)
S01=1
S02=1
M=100000
n=100
dw1=rn.randn(M,n)
dw2=rn.randn(M,n)
S1=S01*np.ones((M,n+1))
S2=S02*np.ones((M,n+1))
B1=0.9
B2=1.1
h=T/n
for i in range(0,n):
    S1[:,i+1]=S1[:,i]*np.exp((r-sigma1**2/2)*h+sigma11*np.sqrt(h)*dw1[:,i]+sigma12*np.sqrt(h)*dw2[:,i])
    S2[:,i+1]=S2[:,i]*np.exp((r-sigma2**2/2)*h+sigma21*np.sqrt(h)*dw1[:,i]+sigma22*np.sqrt(h)*dw2[:,i])

x=np.exp(-r*T)*(np.max(S2,1)<B2)*(np.min(S1,1)>B1)*(S1[:,-1]+S2[:,-1])

V=[np.mean(x),np.std(x)/np.sqrt(M)]
print("V=", V)
