"""
1א
"""
import numpy as np
from numpy import random as rn


X0=1.5
T=1
N=100
h=T/N
M=10000
s=0.3
dw=np.sqrt(h)*rn.randn(M,N)
X=X0*np.ones((M,N+1))
for i in range(0,N):
    X[:,i+1]=X[:,i]+X[:,i]*(X[:,i]**2-4)*h+s*dw[:,i]
pX=(X[:,-1]>0)
print("א.")
print("EX1=",np.mean(X[:,-1]),np.std(X[:,-1])/np.sqrt(M))
print("p(X1>0)=",np.mean(pX),np.std(pX)/np.sqrt(M))

"""
1ב
"""
X0=1.8
T=1
N=100
h=T/N
M=10000
s=0.3
dw=np.sqrt(h)*rn.randn(M,N)
X=X0*np.ones((M,N+1))
for i in range(0,N):
    X[:,i+1]=X[:,i]+X[:,i]*(X[:,i]**2-4)*h+s*dw[:,i]
pX=(X[:,-1]>0)
print("ב.")
print("EX1=",np.mean(X[:,-1]),np.std(X[:,-1])/np.sqrt(M))
print("p(X1>0)=",np.mean(pX),np.std(pX)/np.sqrt(M))


