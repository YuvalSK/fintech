
# coding: utf-8

# Exercise Set 1, Question 5

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn


# In[3]:

get_ipython().magic('matplotlib inline')


# In[ ]:

g=0
N=100000
for i in range(0,N): 
    x=-2*rn.rand(1,10)+1
    if (np.sum((x**2))<1):
        #print("hey")
        g+=1

volume = 2**10*g/N
error = 2**10*np.sqrt((g/N-(g/N)**2)/N)

