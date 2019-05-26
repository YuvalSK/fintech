"""
מבחן תשע"ד מועד א' שאלה 1
"""
import numpy as np
from numpy import random as rn
import matplotlib
import matplotlib.pyplot as plt

s=0.25 #sigma
X0=1
Y0=0
rho = np.linspace(-1,1,21)
t=2
n=100
h=t/n
M = 10000
mean_p=np.zeros(np.shape(rho)) #mean price as rho function
var_p=np.zeros(np.shape(rho))  #var price as rho function
z1=rn.randn(M,n)*np.sqrt(h) #N(0,h)
z2=rn.randn(M,n)*np.sqrt(h)
X=X0*np.ones((M,n+1))
Y=Y0*np.ones((M,n+1))
for i in range(0,len(rho)):
    ro=rho[i]
    for j in range(0,n):
        dw1=z1[:,j]
        dw2=ro*z1[:,j]+np.sqrt(1-ro**2)*z2[:,j]
        X[:,j+1]=X[:,j]+Y[:,j]*h+s*dw1
        Y[:,j+1]=Y[:,j]-X[:,j]*h+s*dw2      
    p=(X[:,-1])**2+(Y[:,-1])**2
    mean_p[i]=np.mean(p)
    var_p[i]=np.std(p)/np.sqrt(M)
print("c(",rho[7],")=",mean_p[7],var_p[7])
print("c(",rho[10],")=",mean_p[10],var_p[10])
print("c(",rho[13],")=",mean_p[13],var_p[13])
plt.plot(rho,mean_p)
plt.xlabel("rho")
plt.ylabel("mean_P")
plt.show()

plt.plot(rho,var_p)
plt.xlabel("rho")
plt.ylabel("var_P")
plt.show()
