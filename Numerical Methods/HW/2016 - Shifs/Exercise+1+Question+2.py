
# coding: utf-8

# Exercise Set 1, Question 2

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn


# In[2]:

get_ipython().magic('matplotlib inline')


# In[33]:

n=100000

# first method 
U=rn.rand(n,12)
N=np.zeros(n)
for i in range (0,n):
    N[i]=sum(U[i,:])-6

plt.hist(N,200)

# second method
x=rn.rand(n,1)
y=rn.rand(n,1)
r=np.sqrt(-2*np.log(x))
theta=2*np.pi*y
u=r*np.cos(theta)
v=r*np.sin(theta)

N=[u,v]
plt.hist(N,200)

#third method
x=np.zeros(n)
y=np.zeros(n)
for i in range(0,n):
    X=2*rn.rand()-1
    Y=2*rn.rand()-1
    while ((X**2+Y**2)>1):
        X=2*rn.rand()-1
        Y=2*rn.rand()-1
    
    x[i]=X
    y[i]=Y


rsq=x**2+y**2

A=np.sqrt(-2*np.log(rsq)/rsq)
u=A*x
v=A*y
N=[u,v]

plt.hist(N,200)

