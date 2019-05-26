"""
תשע"א מועד ב שאלה 5
"""
import numpy as np
from numpy import random as rn

import scipy.stats as ss


T=1
x0=5
y0=0.5*x0**2
a=0.2
gamma=0.3
betta=0.2
M=10000
n=100
h=T/n
Y=y0*np.ones((M,n+1))
Y2=y0*np.ones((M,n+1))
dw=rn.randn(M,n)
   #חישוב ה C
for i in range(0,n):
    Y[:,i+1]=Y[:,i]+((2*a+gamma**2)*Y[:,i]+0.5*betta**2)*h+np.sqrt(h*2*Y[:,i]*(betta**2+gamma**2*2*Y[:,i]))*dw[:,i]+(0.5*betta**2+2*gamma**2*Y[:,i])*(h*dw[:,i]**2-h)
    Y2[:,i+1]=Y2[:,i]+((2*a+gamma**2)*Y2[:,i])*h+np.sqrt(h*2*Y2[:,i]*(gamma**2*2*Y2[:,i]))*dw[:,i]+(2*gamma**2*Y2[:,i])*(h*dw[:,i]**2-h)  #betta=0

q=np.cov(Y[:,-1],Y2[:,-1])
C=-q[0,1]/q[1,1]

M=90000  #סימולציה אמיתית
EYT=0.5*x0**2*np.exp(2*a+gamma**2)*T
Y=y0*np.ones((M,n+1))
Y2=y0*np.ones((M,n+1))
dw=rn.randn(M,n)
for i in range(0,n):
    Y[:,i+1]=Y[:,i]+((2*a+gamma**2)*Y[:,i]+0.5*betta**2)*h+np.sqrt(h*2*Y[:,i]*(betta**2+gamma**2*2*Y[:,i]))*dw[:,i]+(0.5*betta**2+2*gamma**2*Y[:,i])*(h*dw[:,i]**2-h)
    Y2[:,i+1]=Y2[:,i]+((2*a+gamma**2)*Y2[:,i])*h+np.sqrt(h*2*Y2[:,i]*(gamma**2*2*Y2[:,i]))*dw[:,i]+(2*gamma**2*Y2[:,i])*(h*dw[:,i]**2-h)  #betta=0

corrected=Y+C*(Y2-EYT) #תיקון לפי משתנה בקרה
V=[np.mean(corrected),np.std(corrected)/np.sqrt(M)]
print("V=", V)
