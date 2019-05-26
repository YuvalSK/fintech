"""
תשע"ד מועד א' שאלה 2
"""
import numpy as np
from numpy import random as rn


S0=1
a=0.04
r=0.06
M=50000
T=1
sigma=0.2
N=100
dw=rn.randn(M,N)

h=T/N
k=1
S=S0*np.ones((M,N+1))
for i in range(0,N):
    S[:,i+1]=S[:,i]+S[:,i]*r*h+S[:,i]*sigma*np.sqrt(h)*dw[:,i]/(1+a*S[:,i]**2)
X=np.exp(-r*T)*(S[:,-1]>k)*(S[:,-1]-k)
EX=np.mean(X)
VX=np.std(X)/np.sqrt(M)
print("א. p=",EX,VX)


X2=np.exp(-r*T)*(np.max(S,1)>k)*(np.max(S,1)-k)
EX2=np.mean(X2)
VX2=np.std(X2)/np.sqrt(M)
print("ב. p=",EX2,VX2)


X3=np.exp(-r*T)*(np.mean(S,1)>k)*(np.mean(S,1)-k)
EX3=np.mean(X3)
VX3=np.std(X3)/np.sqrt(M)
print("ג. p=",EX3,VX3)
