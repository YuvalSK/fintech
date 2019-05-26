"""
תשע"ו מועד א שאלה 2
"""
import numpy as np
from numpy import random as rn
import matplotlib
import matplotlib.pyplot as plt

"""
א
"""
T=1
r=0.02
sigma1=0.1
sigma2=0.2
S01=10
S02=20
M=50000
#n=252
z1=rn.randn(M)
z2=rn.randn(M)

#h=T/n
k1=11
k2=22
k3=31
rho = np.linspace(-1,1,21)
mean_c12=np.zeros(np.shape(rho))
mean_c3=np.zeros(np.shape(rho))
for i in range(0,len(rho)):
    ro=rho[i]
    S1T=S01*np.exp((r-sigma1**2/2)*T+sigma1*np.sqrt(T)*z1)
    S2T=S02*np.exp((r-sigma2**2/2)*T+sigma2*np.sqrt(T)*(ro*z1+np.sqrt(1-ro**2)*z2))
    c12=np.exp(-r*T)*((S1T-k1)*(S1T>k1)+(S2T-k2)*(S2T>k2))
    c3=np.exp(-r*T)*(S1T+S2T-k3)*(S1T+S2T>k3)
    mean_c12[i]=np.mean(c12)
    mean_c3[i]=np.mean(c3)

plt.plot(rho,mean_c12)
plt.plot(rho,mean_c3)
plt.xlabel("rho")
plt.ylabel("mean_c")
plt.show()

"""
ב
"""
T=1
r=0.02
sigma1=0.1
sigma2=0.2
S01=10
S02=20
M=50000
#n=252
z1=rn.randn(M)
z2=rn.randn(M)

#h=T/n
k1=11
k2=22
k3=np.linspace(25,40,21)
ro = 0.5
mean_c12=np.zeros(np.shape(k3))
mean_c3=np.zeros(np.shape(k3))
for i in range(0,len(k3)):
    S1T=S01*np.exp((r-sigma1**2/2)*T+sigma1*np.sqrt(T)*z1)
    S2T=S02*np.exp((r-sigma2**2/2)*T+sigma2*np.sqrt(T)*(ro*z1+np.sqrt(1-ro**2)*z2))
    c12=np.exp(-r*T)*((S1T-k1)*(S1T>k1)+(S2T-k2)*(S2T>k2))
    c3=np.exp(-r*T)*(S1T+S2T-k3[i])*(S1T+S2T>k3[i])
    mean_c12[i]=np.mean(c12)
    mean_c3[i]=np.mean(c3)
plt.plot(k3,mean_c12)
plt.plot(k3,mean_c3)
plt.xlabel("k3")
plt.ylabel("mean_c")
plt.show()
