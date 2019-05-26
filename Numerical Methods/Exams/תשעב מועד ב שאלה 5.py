"""
תשע"ב מועד ב שאלה 5
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 
import scipy.stats as ss
S0=0.95
a=0.05
r=0.1
c=0.2
M=50000
T=2
b=np.linspace(0,0.1,20)
N=200
dw1=rn.randn(M,N)
dw2=rn.randn(M,N)
h=T/N
k=1
y=[]
z=[]
for x in b:
    S=S0*np.ones((M,N+1))
    for i in range(0,N):
        S[:,i+1]=S[:,i]+S[:,i]*r*h+S[:,i]*(a+x*(S[:,i]-S0)**2)*np.sqrt(h)*dw1[:,i]+S[:,i]*c*np.sqrt(h)*dw2[:,i]
    X=np.exp(-r*T)*(S[:,-1]>k)*(S[:,-1]-k)
    EX=np.mean(X)
    VX=np.std(X)/np.sqrt(M)
    z.append(x)
    y.append(EX)
    
plt.plot(z,y)
plt.xlabel("b")
plt.ylabel("Price")
plt.show()
"""
בדיקה עבור בי =0
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))

def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
def blsprice(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r*T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))

print(blsprice("C",S0, k, r, np.sqrt(a**2+c**2), T))
"""
