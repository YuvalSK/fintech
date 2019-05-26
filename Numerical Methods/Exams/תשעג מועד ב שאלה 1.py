"""
1א
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 

X0=0
T=1
N=100
h=T/N
M=10000
s=0.1
dw=np.sqrt(h)*rn.randn(M,N)
a=np.linspace(0,5,50)
y=[]
z=[]
for x in a:
    X=X0*np.ones((M,N+1))
    for i in range(0,N):
        X[:,i+1]=X[:,i]+x*X[:,i]*(2-X[:,i])*h+s*dw[:,i]
    maxX=np.max(X,1)
    maxX10=(maxX<1)
    z.append(x)
    y.append(np.mean(maxX10))

plt.plot(z,y)
plt.xlabel("a")
plt.ylabel("P")
plt.show()

"""
1ב
"""

X0=0
T=1
N=100
h=T/N
M=10000
s=np.linspace(0,5,50)
a=0.1
y=[]
z=[]
for x in s:
    X=X0*np.ones((M,N+1))
    for i in range(0,N):
        X[:,i+1]=X[:,i]+a*X[:,i]*(2-X[:,i])*h+x*dw[:,i]
    maxX=np.max(X,1)
    maxX10=(maxX<1)
    z.append(x)
    y.append(np.mean(maxX10))

plt.plot(z,y)
plt.xlabel("sigma")
plt.ylabel("P")
plt.show()
