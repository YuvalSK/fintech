"""
Q5
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt

X0=1
s=0.3
a=1
A=5
h=0.1
M=10000
T=2
N=int(T/h)
X=X0*np.ones((M,N+1))
dw=np.sqrt(h)*rn.randn(M,N)

#calculate for h=0.1
for i in range(0,N):
    # Eq. for the process - EM Method
    X[:,i+1]=X[:,i]+a*(A-X[:,i])*h+s*np.sqrt(X[:,i])*dw[:,i]

EX=np.mean(X[:,-1]) #E(x)
VX=np.std(X[:,-1])/np.sqrt(M) # V(x)
se=VX/np.sqrt(M) #stochastic error

print("h=0.1:","\n------","\nE[X]=",EX,"\nV[X]=",VX,"\nstochastic error=",se)

#calculate for diffrent h

X0=1
s=0.3
a=1
A=5
M=100000
T=2
h=np.linspace(0.1,0.3,10)
dw=rn.randn(M,N)
Err = []

for x in h:
    N=int(T/x)
    X=X0*np.ones((M,N+1))
    dw2=np.sqrt(x)*dw
    for i in range(0,N):
        X[:,i+1]=X[:,i]+a*(A-X[:,i])*x+s*np.sqrt(X[:,i])*dw2[:,i]
    EX=np.mean(X[:,-1])
    VX=np.std(X[:,-1])/np.sqrt(M)
    se=VX/np.sqrt(M) #stochastic error
    Err.append(VX)
    print("\nh=:",x,"\n------","\nE[X]=",EX,"\nV[X]=",VX,"\nstochastic error=",se)

'''
#plt the Error as a Function of h
plt.plot(h, Err)
plt.show()
'''