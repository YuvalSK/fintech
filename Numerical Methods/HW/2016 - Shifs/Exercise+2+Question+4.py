
# coding: utf-8

# Exercise Set 2, Question 4

# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn

# In[111]:


sigma=np.linspace(0.1, 0.6,(0.6-0.1)/0.02+1)
prob=np.zeros(len(sigma))

Tmax=10
N=200
h=Tmax/N
t=np.linspace(0, Tmax,Tmax/h+1)

M=10000
Z=rn.randn(M,N)

for k in range(0,len(sigma)-1):
    X=(np.zeros((M,N+1)))
    X[:,0]=X[:,0]+0.5
    sig=sigma[k]

    for j in range(0,M):
        for i in range(0,N):
            if (np.abs(X[j,i])<10):
                X[j,i+1]=X[j,i]+h*X[j,i]*(1-X[j,i])+sig*np.sqrt(h)*Z[j,i];
            else:
                X[j,i+1]=10;
            
        
    
    
    prob[k]=np.sum(X[:,N]==10)/M;
    


plt.plot(sigma[:-1],prob[:-1])
#We expect the probability of divergence to increase with sigma - 
#higher fluctuations gives more chance of divergence. 


# In[116]:

plt.plot(sigma[:-1],prob[:-1])


# In[92]:


X=np.concatenate( X, axis=0 )
plt.show()

