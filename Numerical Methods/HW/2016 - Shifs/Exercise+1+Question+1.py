# coding: utf-8
# Exercise Set 1, Question 1
#המטרה למצוע מה מקדם המתאם של 2 מספרים נורמליים בין ערכים קבועים מראש

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn

'''
#shifs HW1 solution
X = np.ones(100000)
for i in range(1,len(X)):
    X[i] = (8121*X[i-1]+28411)%134456
Y=X/134456
plt.hist(Y,100)
rho=np.zeros(100)
for j in range(0,99):
    rho[j]=np.corrcoef(X[0:99900],X[(1+j):(99901+j)])[1,0]
print(rho)
mu=np.zeros(100)
for j in range(0,99):
    mu[j]=np.corrcoef(X[1:98900],X[(1001+j):(99900+j)])[1,0]

print(mu)
'''

M=21
c=0
a=69
N=20
y=np.ones(M-1)

y = (x*a +c )%M

print(y.corr())