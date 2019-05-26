"""
תשע"ה מועד א' שאלה 1
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 


s=0.35
c=1
M=50000
T=2
N=200
dw1=rn.randn(M,N)
X0=-1+2*rn.rand(M)
h=T/N
X=np.ones((M,N+1))
X[:,0]=X0
for i in range(0,N):
    X[:,i+1]=X[:,i]-c*X[:,i]*h*(X[:,i]**2-1)+s*dw1[:,i]*np.sqrt(h)/(1+X[:,i]**2)
EX=np.mean(X[:,-1])
VX=np.var(X[:,-1])
print("EX2=",EX)
print("VX2=",VX)
    
plt.hist(X[:,-1],bins=1000, normed=1)
plt.show()
