
# coding: utf-8

# In[ ]:

Exercise Set 1, Question 3


# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn
import sympy


# In[23]:

get_ipython().magic('matplotlib inline')


# In[2]:

u = rn.rand()
x = sympy.symbols('x')
equation = sympy.Eq(u, 15/8*(x - 2*x**3/3 + x**5/5) )


# In[3]:

sympy.solve(equation)[0]


# In[20]:

def tfm():
    u = rn.rand()
    #x = fzero( @(x) 15/8*(x - 2*x**3/3 + x**5/5) - u  ,[0,1]) # solve the equation F(x)=u 
    x = sympy.symbols('x')
    equation = sympy.Eq(u, 15/8*(x - 2*x**3/3 + x**5/5) )
    sympy.solve(equation)[0] # output: [1]
    return(x)


#Rejection method from uniform distribution on [0,1]:

def rej():

    s=0;   # flag

    while s==0:
       x = rn.rand(1)
       u = 15/8*rn.rand(1);
       if u < 15/8*(1-x**2)**2:
           s=1;
    return(x)


#Rejection method from 20 level Ziggurat:

def zig(x):
                 #X - axis           #Y - axis
    X=np.array([[               0,   1.875000000000000],  #Zero floor
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
                [1.000000000000000,                   0]])   #20th floor
    s=0 # flag

    while s==0:
        u1 = rn.rand()             #use to choose Ziggurat level and the number in this level
        n  = np.ceil(20*u1)         #Ziggurat level
        x  = X[n+1,1]*(1+20*u1-n)   #in the level 
        if x<X[n,0]:                #in this case certainly keep
            s=1
        else:                       # in this case need to test whether to reject
            u2 = rand(1);
            if (15/8*(1-x**2)**2 > X[n+1,1]+u2*(X[n,1]-X[n+1,1])):
                s=1
    


#For each method can run, say, 50000 times and make a histogram of the results, using code like this:
M = np.zeros(50000)
for i in range(0,50000):
    M[i]=rej()[0]

plt.hist(M)

x=0:0.005:1;
y=15/8*(1-x**2)**2
plt.plot(x,500*y,'r')


# In[25]:

plt.hist(M,100)


# In[ ]:



