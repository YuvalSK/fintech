"""
תשע"ג מועד א' 1
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 

S0=1
a=0.3
r=0.1
b=0.3
M=50000
T=1
c=np.linspace(0,1,30)
N=200
dw1=rn.randn(M,N)
dw2=rn.randn(M,N)
h=T/N
k=1.1
y=[]
z=[]
Y=[]
for x in c:
    S=S0*np.ones((M,N+1))
    for i in range(0,N):
        S[:,i+1]=S[:,i]+S[:,i]*r*h+S[:,i]*a*np.sqrt(h)*dw1[:,i]+S[:,i]*b*np.sqrt(h)*dw2[:,i]/(1+x*S[:,i]**2)
    X=np.exp(-r*T)*(S[:,-1]>k)*(S[:,-1]-k)
    EX=np.mean(X)
    error=np.std(X)/np.sqrt(M)
    z.append(x)
    y.append(EX)
    Y.append(error)
    
plt.plot(z,y)
plt.xlabel("c")
plt.ylabel("Price")
plt.show()
plt.plot(z,Y)
plt.xlabel("c")
plt.ylabel("error")
plt.show()
