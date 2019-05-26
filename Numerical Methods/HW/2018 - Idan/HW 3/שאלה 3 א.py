"""
Q3A
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt

r=0.1
s=0.4  #סיגמא
S0=1
#scale to plot
k=np.linspace(0.95,1.6,100)
T=60/252
M=10000
n=600
S=S0*np.ones((M,n+1))
dw=rn.randn(M,n)
h=T/n
p=[]
err=[]

for x in k:
    for i in range(0,n):
        S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])
    mm=np.max(S,1)-x  #התגמול עבור k
    mm0=(mm>0) #נותן 1 כשהתגמול גדול מ0 ו 0 אחרת
    p.append(np.mean(mm0))  #ההסתברות
    err.append(np.std(mm0)/np.sqrt(M))


plt.plot(k,p)
plt.xlabel("k")
plt.ylabel("p")
#plt.show()

plt.plot(k,err)
plt.xlabel("k")
plt.ylabel("err")
#plt.show()
