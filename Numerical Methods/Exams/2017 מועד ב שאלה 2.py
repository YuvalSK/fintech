import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 

mu=0.05
T=1
N=100
h=T/N
M=10000
S0=1.1
dw=np.sqrt(h)*rn.randn(M,N)
sigma=np.linspace(0.01,10,51)
y=[]
z=[]
q=[]
for x in sigma:
    S=S0*np.ones((M,N+1))
    for i in range(0,N):
        S[:,i+1]=S[:,i]*np.exp((mu-x**2/2)*h+x*dw[:,i]) 
    minS=np.min(S,1)
    maxS=np.max(S,1)
    minmax=(minS>0.5*S0)*(maxS<1.5*S0)
    z.append(x)
    y.append(np.mean(minmax))
    q.append(np.std(minmax)/np.sqrt(M))

plt.plot(z,y)
plt.xlabel("sigma")
plt.ylabel("P")
plt.show()
