"""
Q1B
"""
import numpy as np
from numpy import random as rn
import matplotlib
import matplotlib.pyplot as plt

def max_s1s2_Eprice(k1,k2,z,rho,M,t):   #תוחלת המחיר
    s1=0.0001 #sigma
    r1=0.0001
    r2=0.0001
    s2=0.0002
    S01=1
    S02=1
    mean_p=np.zeros(np.shape(rho)) #mean price as rho function

    for i in range(0,len(rho)):
        # To keep the dependence between Brownian motion - Z1,Z2 ~ N(0,h)
        ## dw1 = z
        ## dw2 = rho(z1) + sqrt(1-rho ^2)*z2
        dw1=z[:,0]
        dw2=rho[i]*z[:,0]+np.sqrt(1-rho[i]**2)*z[:,1]

        S1_100=S01*np.exp((r1-s1**2/2)*t+s1*dw1) #S1 price when t=100
        S2_100=S02*np.exp((r2-s2**2/2)*t+s2*dw2) #S2 price when t=100

        # today value of insurance, two conditions: (1) call option for both (2) max of values
        p=np.exp(-r1*t)*((S1_100-k1)*(S2_100-k2<S1_100-k1)*(S2_100>k2)*(S1_100>k1)+(S2_100-k2)*(S2_100-k2>S1_100-k1)*(S2_100>k2)*(S1_100>k1))

        mean_p[i]=np.mean(p)
        # price for specific rho
    return mean_p


def max_s1s2_VARprice(k1,k2,z,rho,M,t):   #פונקציה בשביל השגיאה
    s1=0.0001 #sigma
    r1=0.0001
    r2=0.0001
    s2=0.0002
    S01=1
    S02=1
    var_p=np.zeros(np.shape(rho))  #var price as rho function

    for i in range(0,len(rho)):
        dw1=z[:,0]
        dw2=rho[i]*z[:,0]+np.sqrt(1-rho[i]**2)*z[:,1]
        S1100=S01*np.exp((r1-s1**2/2)*t+s1*dw1)       #S1 price when t=100
        S2100=S02*np.exp((r2-s2**2/2)*t+s2*dw2)       #S2 price when t=100
        p=np.exp(-r1*t)*((S1100-k1)*(S2100-k2<S1100-k1)*(S2100>k2)*(S1100>k1)+(S2100-k2)*(S2100-k2>S1100-k1)*(S2100>k2)*(S1100>k1))
        var_p[i]=np.std(p)/np.sqrt(M)
    return var_p


#graph:
M=10000
t=100
z=rn.randn(M,2)*np.sqrt(t) #N(0,t)
rho = np.linspace(-1,1,50)

plt.plot(rho,max_s1s2_Eprice(1.0001,1.0002,z,rho,M,t),label="k1=1.0001,k2=1.0002")
plt.plot(rho,max_s1s2_Eprice(1.0001,0.9999,z,rho,M,t),label="k1=1.0001,k2=0.9999")
plt.plot(rho,max_s1s2_Eprice(0.9999,0.9998,z,rho,M,t),label="k1=0.9999, k2=0.9998")
plt.xlabel("rho")
plt.ylabel("mean_P")
plt.legend()
plt.grid()
plt.show()




