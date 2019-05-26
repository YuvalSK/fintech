"""
תשע"ב מועד א' שאלה 5
"""
import numpy as np
from numpy import random as rn

import scipy.stats as ss

#Black and Scholes
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))

def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
def blsprice(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))

T=1
n=100
h=T/n
a=0.03
b=0.09
r=0.1
rho=0.1
S0=0.9
v0=0.04
k=1
M=1000
dw1=rn.randn(M,n)
dw2=rn.randn(M,n)
S1=S0*np.ones((M,n+1))
S2=S0*np.ones((M,n+1))  #משתנה בקרה
v=v0*np.ones((M,n+1))
   #חישוב ה C
for i in range(0,n):
    v[:,i+1]=v[:,i]+a*(v0-v[:,i])*h+b*np.sqrt(h*np.abs(v[:,i]))*(rho*dw1[:,i]+np.sqrt(1-rho**2)*dw2[:,i])
    S1[:,i+1]=S1[:,i]*np.exp((r-v[:,i]/2)*h+np.sqrt(np.abs(v[:,i])*h)*dw1[:,i])
    S2[:,i+1]=S2[:,i]*np.exp((r-v0/2)*h+np.sqrt(v0*h)*dw1[:,i])
x=(S1[:,-1]-k)*(S1[:,-1]>k)*np.exp(-r*T)   #האופציה המבוקשת
y=np.exp(-r*T)*(S2[:,-1]>k)*(S2[:,-1]-k)  # אופציה רגילה קרובה עם משתנה בקרה
q=np.cov(x,y)
C=-q[0,1]/q[1,1]

M=10000  #סימולציה אמיתית
V_C=blsprice("C",S0, k, r, np.sqrt(v0), T)
dw1=rn.randn(M,n)
dw2=rn.randn(M,n)
S1=S0*np.ones((M,n+1))
S2=S0*np.ones((M,n+1))  #משתנה בקרה
v=v0*np.ones((M,n+1))
   #חישוב ה C
for i in range(0,n):
    v[:,i+1]=v[:,i]+a*(v0-v[:,i])*h+b*np.sqrt(h*np.abs(v[:,i]))*(rho*dw1[:,i]+np.sqrt(1-rho**2)*dw2[:,i])
    S1[:,i+1]=S1[:,i]*np.exp((r-v[:,i]/2)*h+np.sqrt(np.abs(v[:,i])*h)*dw1[:,i])
    S2[:,i+1]=S2[:,i]*np.exp((r-v0/2)*h+np.sqrt(v0*h)*dw1[:,i])
x=(S1[:,-1]-k)*(S1[:,-1]>k)*np.exp(-r*T)   #האופציה המבוקשת
y=np.exp(-r*T)*(S2[:,-1]>k)*(S2[:,-1]-k)  # אופציה רגילה קרובה עם משתנה בקרה
corrected=x+C*(y-V_C) #תיקון לפי משתנה בקרה
V=[np.mean(corrected),np.std(corrected)/np.sqrt(M)]
print("V=", V)
