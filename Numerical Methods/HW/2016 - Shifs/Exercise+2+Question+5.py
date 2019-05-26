# coding: utf-8
# Exercise Set 2, Question 5

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn

# In[27]:

def w(M,N):

    # to find E[X(2)]
    # using M simulations and N subdivisions of [0,2] 

    a=5
    alpha=1
    sigma=0.3
    X0=1
    T=2
    h=T/N

    r=rn.randn(M,N)    # normal random numbers 

    X=X0*np.ones(M,) 
    for i in range(0,N):
        # solving   dX = alpha(a-X)dt + sqrt(X)sigma dW
        X = X+h*alpha*(a-X)+np.sqrt(h)*sigma*np.sqrt(X)*r[:,i]
    

    a1=np.mean(X)
    a2=np.std(X)
    return(a1,a2)


a,b = w(10000,20)

print("Mean: {0} \nstd: {1}".format(a,b))
#gave answers a=4.5221 and b=0.4337. The stochastic error is plus or minus 0.004. Further runs gave a=4.5184, 4.5159, 4.5147, 4.5060, 4.5113, 4.5168.
#To implement for different values of h between 0.1 and 0.3 I did the following:

h=np.linspace(0.1, 0.3,(0.3-0.1)/0.01+1)
hs=np.zeros(len(h));
As=np.zeros(len(h));
bs=np.zeros(len(h));

for i in range(0,len(h)):
    N=int(np.floor(2/h[i]))
    As[i], bs[i]=w(100000,N)
    hs[i]=2/N


#The b's varied little: from 0.4383 to 0.4744, giving a stochastic error of about plus or minus 0.0015.

plt.plot(hs,As)
plt.show()

