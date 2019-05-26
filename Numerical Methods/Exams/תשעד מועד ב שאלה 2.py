"""
1א
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 

S0=1
k=S0
r=0.02
T=1
N=100
h=T/N
M=10000
dw=np.sqrt(h)*rn.randn(M,N)
s=np.linspace(0,5,50)
B=0.8*S0
y=[]
z=[]
Y=[]
for x in s:
    S=S0*np.ones((M,N+1))
    for i in range(0,N):
        S[:,i+1]=S[:,i]*np.exp((r-x**2/2)*h+x*dw[:,i])
    P=(np.min(S,1)<B)*(S[:,-1]-k)*(S[:,-1]>k)*np.exp(-r*T)
    z.append(x)
    y.append(np.mean(P))
    Y.append(np.std(P)/np.sqrt(M))

plt.plot(z,y)
plt.xlabel("sigma")
plt.ylabel("P")
plt.show()
plt.plot(z,Y)
plt.xlabel("sigma")
plt.ylabel("error")
plt.show()

"""
1ב
"""

"""
S0=1
k=S0
r=0.02
T=1
N=100
h=T/N
M=10000
dw=np.sqrt(h)*rn.randn(M,N)
s=np.linspace(0,5,50)
"""
B=0.7*S0
y=[]
z=[]
Y=[]
for x in s:
    S=S0*np.ones((M,N+1))
    for i in range(0,N):
        S[:,i+1]=S[:,i]*np.exp((r-x**2/2)*h+x*dw[:,i])
    P=(np.min(S,1)<B)*(S[:,-1]-k)*(S[:,-1]>k)*np.exp(-r*T)
    z.append(x)
    y.append(np.mean(P))
    Y.append(np.std(P)/np.sqrt(M))

plt.plot(z,y)
plt.xlabel("sigma")
plt.ylabel("P")
plt.show()
plt.plot(z,Y)
plt.xlabel("sigma")
plt.ylabel("error")
plt.show()
