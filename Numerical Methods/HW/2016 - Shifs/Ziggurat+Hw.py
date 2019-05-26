# coding: utf-8
# # Ziggurat
# ## Rejection method from 20 level Ziggurat: שיטת הדחייה עבור זיגורט של 20 רמות

#import packages
import numpy as np
from numpy import random as rn

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

# In[224]:

def zig():
                   #X - axis           #Y - axis
    X=np.array([[               0,   1.875000000000000],  #Level Zero
               [0.243700966784892,   1.658900379046860],
               [0.326107066633865,   1.497408366341702],
               [0.384904660677814,   1.360585688083304],
               [0.432611507516508,   1.238851324495972],
               [0.473788396832940,   1.127696878966890],
               [0.510671893321394,   1.024570610932918],
               [0.544552878330669,   0.927860651912006],
               [0.576262880271060,   0.836472349192757],
               [0.606385282511440,   0.749623792562287],
               [0.635361459557960,   0.666736035119904],
               [0.663551251394990,   0.587369616267762],
               [0.691272577868383,   0.511185939619582],
               [0.718832774671270,   0.437923160843028],
               [0.746560490929985,   0.367381406775015],
               [0.774848504999009,   0.299414976939616],
               [0.804227184885832,   0.233931382828638],
               [0.835520550158683,   0.170900393898353],
               [0.870266300345544,   0.110385943864877],
               [0.912363601081001,   0.052663686548078],
               [1.000000000000000,                   0]])   #level 20

    s=0    # flag

    while (s==0):
        u1 = rn.rand(1)                # use to choose Ziggurat level and the number in this level
        n  = int(np.ceil(19*u1))       # Ziggurat level
        x  = X[n+1,0]*(1+19*u1-n)      # number in the level 
        if (x<X[n,0]):                 # in this case certainly keep
            s=1
        else:                           # in this case need to test whether to reject
            u2 = rn.rand(1)
            if (15/8*(1-x**2)**2 > X[n+1,1]+u2*(X[n,1]-X[n+1,1])):
                s=1

    return(x)

#For each method can run, say, 50000 times and make a histogram of the results, using code like this:
M = np.zeros(50000)
n = 50000             #num of simulations
for i in range(0,n):
    M[i]=zig()

Axis = plt.hist(M,100)   #plot hist & saving axis values
Ym = Axis[0]
Ym = Ym*100 / n  
Xm = Axis[1][1:] 

x=np.linspace(0,1,201) #the values are: [0,0.005,0.010...,1]
y=15/8*(1-x**2)**2     #reject CDF

fig, ax = plt.subplots()
ax.plot(x, y, color='red')
ax.scatter(Xm,Ym)
plt.show()
