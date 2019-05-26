"""
תשע"ו מועד ב שאלה 2
"""
import numpy as np
from numpy import random as rn

import scipy.stats as ss

T=1
r=0.01
sigma1=0.1
sigma2=0.2
S01=10
S02=10
M=50000
n=252
z1=rn.randn(M,n)
z2=rn.randn(M,n)
S1=S01*np.ones((M,n+1))
S2=S02*np.ones((M,n+1))
h=T/n
k=20.4
ro=0.4

for i in range(0,n):
    S1[:,i+1]=S1[:,i]*np.exp((r-sigma1**2/2)*h+sigma1*np.sqrt(h)*z1[:,i])
    S2[:,i+1]=S2[:,i]*np.exp((r-sigma2**2/2)*h+sigma2*np.sqrt(h)*(ro*z1[:,i]*i/n+np.sqrt(1-(ro*i/n)**2)*z2[:,i]))

x=np.exp(-r*T)*(S1[:,-1]+S2[:,-1]-k)*(S1[:,-1]+S2[:,-1]>k)
p1=(S1[:,-1]+S2[:,-1]>k)
p2=(x>np.mean(x))
P2=[np.mean(p2),np.std(p2)/np.sqrt(M)]
P1=[np.mean(p1),np.std(p1)/np.sqrt(M)]
V=[np.mean(x),np.std(x)/np.sqrt(M)]
print("א. V=", V)
print("ב. P(בתוך הכסף)=", P1)
print("ג. P(x>V)=", P2)
