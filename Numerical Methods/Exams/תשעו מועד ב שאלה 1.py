"""
תשע"ו מועד ב שאלה 1
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 

X0=0
a=np.linspace(-2,2,21)
s=1.2
M=10000
T=2
N=200
dw1=rn.randn(M,N)
h=T/N
y=[]
z=[]
Y=[]
for x in a:
    X=X0*np.ones((M,N+1))
    for i in range(0,N):
        X[:,i+1]=X[:,i]+X[:,i]*(x-X[:,i]**2)*h+s*np.sqrt(h)*dw1[:,i]/(1+X[:,i]**2)
    EX2=np.mean(X[:,-1])
    error=np.std(X[:,-1])/np.sqrt(M)
    VX2=np.var(X[:,-1])
    Y.append(VX2)
    y.append(EX2)
    z.append(error)
print("a ,EX2, VX2, error")
print(a[0],y[0], Y[0], z[0])
print(a[10],y[10], Y[10], z[10])
print(a[-1],y[-1], Y[-1], z[-1])
plt.plot(a,y)
plt.xlabel("a")
plt.ylabel("EX2")
plt.show()
plt.plot(a,Y)
plt.xlabel("c")
plt.ylabel("VX2")
plt.show()
