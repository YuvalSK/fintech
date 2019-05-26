"""
Q3C
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt

r=0.1
s=0.4  #סיגמא
S0=1
k=1.4
T=60/252
M=10000
n=600
S=S0*np.ones((M,n+1))
dw=rn.randn(M,n)
h=T/n
mu=np.linspace(0,0.16,100)
V=[]     #שווי האופציה 
err=[]

for x in mu:
    newdw=x+dw
    for i in range(0,n):
        S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*newdw[:,i])

    mm=np.max(S,1)
    #Likelyhood Ratio - as in Importance sampling
    LR=np.exp(-x*np.sum(newdw,1))*np.exp(n*x**2/2)
    nn=(mm>k)*(mm-k)*np.exp(-r*T)*LR

    V.append(np.mean(nn))
    err.append(np.std(nn)/np.sqrt(M))

print('optimal mu-{1},min Error-{0})'.format(np.min(err),mu[err.index(np.min(err))]))

#plt.plot(mu,V)
#plt.xlabel("mu")
#plt.ylabel("V")
#plt.show()
plt.plot(mu,err)
plt.xlabel("mu")
plt.ylabel("err")
plt.show()

