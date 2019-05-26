#!/usr/bin/env python3
"""
Q3A
"""
import matplotlib
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve
import numpy as np
from numpy import random as rn
def random_F(N,lamda):
    X=[]
    while len(X)<N:
        u=rn.rand(1)
        x0=np.exp(-lamda)
        if x0 >=u:
            x=0
        else:
            n=1
            Y=lamda*x0/n
            y=lamda*x0/n
            while x0+Y<u:
                n+=1
                y=lamda*y/n
                Y+=y
            x=n
        X.append(x)
    return X
lamda=5
z=random_F(100000,lamda)                   #random poi numbers
print(np.mean(z))
x=np.random.poisson(5, 100000)
plt.hist(x,bins=10, normed=1) 
plt.hist(z,bins=10, normed=1)      #F histogram
plt.show()
