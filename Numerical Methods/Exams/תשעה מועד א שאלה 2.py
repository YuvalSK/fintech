"""
תשע"ה מועד א שאלה 2
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 

S0=1
r=0.01
M=10000
T=1
sigma=0.2
N=252
h=T/N
dw=rn.randn(M,N)*np.sqrt(h)
k=np.linspace(0.8,1.2,19)
Y=[]
for x in k:
    S=S0*np.ones((M,N+1))
    for i in range(0,N):
        S[:,i+1]=S[:,i]*np.exp((r-sigma**2/2)*h+sigma*dw[:,i])
    X1=np.exp(-r*T)*(np.max(S,1)>x)*(np.max(S,1)-x)
    X2=np.exp(-r*T)*(np.min(S,1)<x)*(x-np.min(S,1))
    Y.append(float(np.corrcoef([X1,X2])[1,0]))

print("rho(k=",k[9],")=",Y[9])

plt.plot(k,Y)
plt.xlabel("k")
plt.ylabel("rho")
plt.show()
