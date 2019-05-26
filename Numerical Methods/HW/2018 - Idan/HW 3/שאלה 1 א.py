"""
Q1A
"""
import numpy as np
from numpy import random as rn
import matplotlib
import matplotlib.pyplot as plt

s1=0.0001 #sigma
r1=0.0001
r2=0.0001
s2=0.0002
S01=1
S02=1
rho = np.linspace(-1,1,50)
t=100
M = 50**2
mean_p=np.zeros(np.shape(rho)) #mean price as rho function
var_p=np.zeros(np.shape(rho))  #var price as rho function
z=rn.randn(M,2)*np.sqrt(t) #N(0,t) to EM keeping the Wt dependence!!!

for i in range(0,len(rho)):

    #To keep the dependence between Brownian motion - Z1,Z2 ~ N(0,h)
    ## dw1 = z
    ## dw2 = rho(z1) + sqrt(1-rho ^2)*z2
    dw1=z[:,0]
    dw2=rho[i]*z[:,0]+np.sqrt(1-rho[i]**2)*z[:,1]

    S1_100=S01*np.exp((r1-s1**2/2)*t+s1*dw1)       #S1 price for t=100
    S2_100=S02*np.exp((r2-s2**2/2)*t+s2*dw2)       #S2 price for t=100

    p=np.exp(-r1*t)*(S2_100-S1_100)*(S2_100>S1_100)#today value of insurance
    #Monta Carlo
    mean_p[i]=np.mean(p) #E(x) - price for specific rho
    var_p[i]=np.std(p)/np.sqrt(M) #std

#MC so to mention the Stochastic Error - print(var_p)

plt.plot(rho,mean_p)
plt.xlabel("rho")
plt.ylabel("mean_P")
plt.title('Price(rho)')
plt.show()

'''
#it's the price but what about the error?
plt.clf()

plt.plot(rho,var_p)
plt.xlabel("rho")
plt.ylabel("var_P")
plt.show()
'''