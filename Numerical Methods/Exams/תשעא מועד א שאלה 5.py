"""
תשע"א מועד א' שאלה 5
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

T=2
r=0.05
sigma11=0.2
sigma12=0.1
sigma21=-0.1
sigma22=0.4
sigma1=np.sqrt(sigma12**2+sigma11**2)
sigma2=np.sqrt(sigma21**2+sigma22**2)
S01=1.2
S02=1
k1=1.2
k2=2
M=10000
dw1=rn.randn(M,1)
dw2=rn.randn(M,1)
S1=S01*np.ones((M,1))
S2=S02*np.ones((M,1))
   #חישוב ה C
S1[:,0]=S01*np.exp((r-sigma1**2/2)*T+sigma11*np.sqrt(T)*dw1[:,0]+sigma12*np.sqrt(T)*dw2[:,0])
S2[:,0]=S02*np.exp((r-sigma2**2/2)*T+sigma21*np.sqrt(T)*dw1[:,0]+sigma22*np.sqrt(T)*dw2[:,0])
x=(S1[:,0]-k1>S2[:,0]-k2)*(S1[:,0]-k1)*(S1[:,0]>k1)*np.exp(-r*T)+(S1[:,0]-k1<S2[:,0]-k2)*(S2[:,0]-k2)*np.exp(-r*T)*(S2[:,0]>k2)   #האופציה המבוקשת
y=np.exp(-r*T)*(S1[:,0]>k1)*(S1[:,0]-k1)  # אופציה רגילה קרובה עם משתנה בקרה
q=np.cov(x,y)
C=-q[0,1]/q[1,1]

M=90000  #סימולציה אמיתית
V_C=blsprice("C",S01, k1, r, sigma1, T)
dw1=rn.randn(M,1)
dw2=rn.randn(M,1)
S1=S01*np.ones((M,1))
S2=S02*np.ones((M,1))
S1[:,0]=S01*np.exp((r-sigma1**2/2)*T+sigma11*np.sqrt(T)*dw1[:,0]+sigma12*np.sqrt(T)*dw2[:,0])
S2[:,0]=S02*np.exp((r-sigma2**2/2)*T+sigma21*np.sqrt(T)*dw1[:,0]+sigma22*np.sqrt(T)*dw2[:,0])
x=(S1[:,0]-k1>S2[:,0]-k2)*(S1[:,0]-k1)*(S1[:,0]>k1)*np.exp(-r*T)+(S1[:,0]-k1<S2[:,0]-k2)*(S2[:,0]-k2)*np.exp(-r*T)*(S2[:,0]>k2)   #האופציה המבוקשת
y=np.exp(-r*T)*(S1[:,0]>k1)*(S1[:,0]-k1)  # אופציה רגילה קרובה עם משתנה בקרה
corrected=x+C*(y-V_C) #תיקון לפי משתנה בקרה
V=[np.mean(corrected),np.std(corrected)/np.sqrt(M)]
print("V=", V)
