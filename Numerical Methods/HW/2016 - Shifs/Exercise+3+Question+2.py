
# coding: utf-8

# Exercise Set 3, Question 2

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn

# Part a & b

# In[17]:

r=0.1
s=0.4
S0=1.0
K=1.1
T=60/252
N=120         # for twice a day observation
#N=600        # for ten times a day observation
M=50000       # number of simulations
h = T/N
Z = rn.randn(M,N)
S = S0*np.ones((M,N+1))

# make prices
for i in range(0,N):
    S[:,i+1]=S[:,i]*np.exp( (r-s**2/2)*h + np.sqrt(h)*s*Z[:,i])


Smax = np.max(S,axis=1)
ret = np.exp(-r*T)*(Smax-K)*(Smax>K)
print(np.mean(ret))
print(np.std(ret)/np.sqrt(M))
#Results of ten runs for the N=120 case: 
#with stochastic error always 0.0006. 


# Part c:

# In[23]:

r=0.1
s=0.4
S0=1.0
K=1.1
T=60/252
N=600        
M=50000      
h = T/N
Z = rn.randn(M,N)
eps = 1e-5 

# we are going to make two matrices of prices
Splus  = (S0+eps)*np.ones((M,N+1))
Sminus = (S0-eps)*np.ones((M,N+1))

# make prices
for i in range(0,N):
    Splus[:,i+1]  = Splus[:,i]  *np.exp( (r-s**2/2)*h + np.sqrt(h)*s*Z[:,i]);
    Sminus[:,i+1] = Sminus[:,i] *np.exp( (r-s**2/2)*h + np.sqrt(h)*s*Z[:,i]);


Splusmax  = np.max(Splus,1);
Sminusmax = np.max(Sminus,1);
retplus  = np.exp(-r*T)*(Splusmax-K)*(Splusmax>K); 
retminus = np.exp(-r*T)*(Sminusmax-K)*(Splusmax>K);
sens     = (retplus-retminus)/(2*eps);
print(np.mean(sens))
print(np.std(sens)/np.sqrt(M))


# Part d:

# In[29]:

r=0.1
s=0.4
S0=1.0
K=1.1
T=60/252
N=600      
M=50000
h = T/N
Z = rn.randn(M,N)
eps = 1e-5; 

# we are going to make two matrices of prices
Splus  = S0*np.ones((M,N+1))
Sminus = S0*np.ones((M,N+1))

# make prices
for i in range(0,N):
    Splus[:,i+1]  = Splus[:,i]  *np.exp( (r-(s+eps)**2/2)*h + np.sqrt(h)*(s+eps)*Z[:,i]);
    Sminus[:,i+1] = Sminus[:,i] *np.exp( (r-(s-eps)**2/2)*h + np.sqrt(h)*(s-eps)*Z[:,i]);


Splusmax  = np.max(Splus,1);
Sminusmax = np.max(Sminus,1);
retplus  = np.exp(-r*T)*(Splusmax-K)*(Splusmax>K); 
retminus = np.exp(-r*T)*(Sminusmax-K)*(Splusmax>K);
sens     = (retplus-retminus)/(2*eps);
print(np.mean(sens))
print(np.std(sens)/np.sqrt(M))


# In[ ]:

#Ran to get expected value of derivative 0.3857 with error 0.0020. Quite reasonable, 
#consistent with the finding in parts a and b that the option value is about 0.1 (with sigma = 0.4). 

