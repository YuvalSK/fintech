"""
תשע"ה מועד ב' שאלה 2
"""
import numpy as np
from numpy import random as rn

import scipy.stats as ss

T=1
r=0.02
sigma1=0.15
sigma2=0.08
S01=1
S02=1
M=50000
n=252
z1=rn.randn(M,n)
z2=rn.randn(M,n)
S1=S01*np.ones((M,n+1))
S2=S02*np.ones((M,n+1))
B=0.9
h=T/n
k=1.1
ro=0.4

for i in range(0,n):
    S1[:,i+1]=S1[:,i]*np.exp((r-sigma1**2/2)*h+sigma1*np.sqrt(h)*z1[:,i])
    S2[:,i+1]=S2[:,i]*np.exp((r-sigma2**2/2)*h+sigma2*np.sqrt(h)*(ro*z1[:,i]+np.sqrt(1-ro**2)*z2[:,i]))

x=np.exp(-r*T)*(np.min(S2,1)>B)*(S1[:,-1]-k)*(S1[:,-1]>k)

V=[np.mean(x),np.std(x)/np.sqrt(M)]
print("V=", V)
