"""
Q4A
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

r=0.1
s=0.4  #סיגמא
S0=1
k=1.1
T=60/252
M=50000
n=60 #60 days, only the last 10 counts!
S=S0*np.ones((M,n+1))
dw=rn.randn(M,n)
h=T/n

for i in range(0,n):
    #GBM with h time periods
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])

mm=np.mean(S[:,51:],1)-k  #ממוצע רק מהיום ה51 עד היום ה60
nn=(mm>0)*mm*np.exp(-r*T) # today's pricing

V=[np.mean(nn),np.std(nn)/np.sqrt(M)] #classic MC

print("א. V=", V)

"""
Q4B
"""

M=5000
dw=rn.randn(M,n)
S=S0*np.ones((M,n+1))
for i in range(0,n):   #חישוב ה C
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])

mm=np.mean(S[:,51:],1)  #ממוצע רק מהיום ה51 עד היום ה60
x=(mm>k)*(mm-k)*np.exp(-r*T)   #האופציה המבוקשת
y=np.exp(-r*T)*(S[:,-1]>k)*(S[:,-1]-k)  # אופציה רגילה קרובה עם משתנה בקרה
q=np.cov(x,y)
C=-q[0,1]/q[1,1]

M=45000  #סימולציה אמיתית
dw=rn.randn(M,n)
S=S0*np.ones((M,n+1))

V_C=blsprice("C",S0, k, r, s, T)

for i in range(0,n):   
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])

mm=np.mean(S[:,51:],1) #ממוצע רק מהיום ה51 עד היום ה60
x=(mm>k)*(mm-k)*np.exp(-r*T)   #האופציה המבוקשת
y=np.exp(-r*T)*(S[:,-1]>k)*(S[:,-1]-k)  # אופציה רגילה קרובה עם משתנה בקרה

corrected=x+C*(y-V_C) #תיקון לפי משתנה בקרה

V=[np.mean(corrected),np.std(corrected)/np.sqrt(M)]
print("ב. V=", V)

"""
Q4C
"""
M=5000
dw=rn.randn(M,n)
S=S0*np.ones((M,n+1))
for i in range(0,n):   #חישוב ה C
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])

mm=np.mean(S[:,51:],1)  #ממוצע רק מהיום ה51 עד היום ה60
x=(mm>k)*(mm-k)*np.exp(-r*T)   #האופציה המבוקשת
y=np.exp(-r*(T-5/252))*(S[:,-6]>k)*(S[:,-6]-k)  # אופציה רגילה קרובה עם משתנה בקרה
q=np.cov(x,y)
C=-q[0,1]/q[1,1]

M=45000  #סימולציה אמיתית
dw=rn.randn(M,n)
S=S0*np.ones((M,n+1))
V_C=blsprice("C",S0, k, r, s, (T-5/252))
for i in range(0,n):   
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])
mm=np.mean(S[:,51:],1) #ממוצע רק מהיום ה51 עד היום ה60
x=(mm>k)*(mm-k)*np.exp(-r*T)   #האופציה המבוקשת
y=np.exp(-r*(T-5/252))*(S[:,-6]>k)*(S[:,-6]-k)  # אופציה רגילה קרובה עם משתנה בקרה
corrected=x+C*(y-V_C) #תיקון לפי משתנה בקרה
V=[np.mean(corrected),np.std(corrected)/np.sqrt(M)]
print("ג. V=", V)

print("תשובה ג עם השגיאה הכי נמוכה לכן המשתנה בקרה הזה עדיף")
