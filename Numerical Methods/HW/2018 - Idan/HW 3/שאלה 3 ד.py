"""
Q3D
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 

r=0.1
s=0.4  #סיגמא
S01=1
epsilon=0.00005
S02=1+epsilon
k=1.4
T=60/252
M=50000
n=600
S1=S01*np.ones((M,n+1))
S2=S02*np.ones((M,n+1))
dw=rn.randn(M,n)
h=T/n

for i in range(0,n):
    S1[:,i+1]=S1[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])
    S2[:,i+1]=S2[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])

mm1=np.max(S1,1)-k
mm2=np.max(S2,1)-k

nn1=(mm1>0)*mm1*np.exp(-r*T)
nn2=(mm2>0)*mm2*np.exp(-r*T)
delta=(nn2-nn1)/epsilon

print("ד. delta(mu=0)=", np.mean(delta),";",np.std(delta)/np.sqrt(M))

r=0.1
s=0.4  #סיגמא
S01=1
k=1.4
T=60/252
M=50000
n=600
h=T/n
epsilon=0.00005
S02=1+epsilon
S1=S01*np.ones((M,n+1))
S2=S02*np.ones((M,n+1))
x=0.07 #mu עם שונות מינמלית לפי סעיף קודם
newdw=x+dw
for i in range(0,n):
    S1[:,i+1]=S1[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*newdw[:,i])
    S2[:,i+1]=S2[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*newdw[:,i])
mm1=np.max(S1,1)-k
mm2=np.max(S2,1)-k
LR=np.exp(-x*np.sum(newdw,1)+n*x**2/2)
nn1=(mm1>0)*mm1*np.exp(-r*T)*LR
nn2=(mm2>0)*mm2*np.exp(-r*T)*LR
delta=(nn2-nn1)/epsilon

print("delta(mu=0.07)=", np.mean(delta),";",np.std(delta)/np.sqrt(M))
print("אפשר לראות שכשמיו שווה ל0.07 ועושים אימפורטנט סמפלינג מגיעים לתוצאה יותר מדוייקת")

