import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 

r0=0.03
T=1
N=100
h=T/N
M=10000
b=0.02
c=0.04
dw=np.sqrt(h)*rn.randn(M,N)
a=np.linspace(0,2,51)
y=[]
z=[]
q=[]
for x in a:
    r=r0*np.ones((M,N+1))
    for i in range(0,N):
        r[:,i+1]=r[:,i]+x*(c-r[:,i])*h+b*dw[:,i]  
    minr=np.min(r,1)
    minrr=(minr<0)
    z.append(x)
    y.append(np.mean(minrr))
    q.append(np.std(minrr)/np.sqrt(M))
print("P(a=",z[25],")=",y[25],q[25])
plt.plot(z,y)
plt.xlabel("a")
plt.ylabel("P")
plt.show()
