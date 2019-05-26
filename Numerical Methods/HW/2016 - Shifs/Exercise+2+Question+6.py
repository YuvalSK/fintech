
# coding: utf-8

# Exercise Set 2, Question 6

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn

# In[26]:

T=5
a=0.15
b=0.1
r=0.05

M=5000
N=1000
h=T/N

A=(a-b**2/2)*h
B=b*np.sqrt(h)
Z=rn.randn(M,N)

valueatend=np.zeros((M,2))   # results of the investment by the 2 strategies

for j in range(0,M):

    S=1
    F=0     # F=0 implies that 75% profit has not yet been acheived 

    for i in range(0,N):
       S=S*np.exp(A+B*Z[j,i])
       if (S>=1.75 and F==0):
         F=S*np.exp(r*T*(1-i/N))
       
    

    if F==0:
       F=S
    
    valueatend[j,0]=F
    valueatend[j,1]=S



print(np.mean(valueatend[:,0]) , np.mean(valueatend[:,1]))
print(np.std(valueatend[:,0]) , np.std(valueatend[:,1]))

#Answers (reproduced in several runs) gave the expected value of 
#the "sell high" strategy to be about 1.83 and the expected value of the 
#"hold 5 years" strategy to be about 2.12 
#(the exact value for this, from question 3, is exp(0.75)=2.1170). 
#BUT the standard deviation of the first strategy is only about 0.18 while for the second strategy 
#it is 0.48. Thus, as usual in these things, the "cost" of a strategy 
#with a high average return is high risk. 
#It is interesting to look at the distributions of the returns from the 2 strategies, 
#beyond just their means and variances, to decide what is relevant for an investment decision. 

