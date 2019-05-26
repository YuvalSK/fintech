#!/usr/bin/env python3
"""
תרגיל 1 שאלה 3 סעיף ב בתרגילים של שיף
"""
import matplotlib
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve
import numpy as np
from numpy import random as rn
"""
ניקח את סי להיות 15/8. הצפיפות של וואי שווה ל1 והצפיפות של איקס נתונה
זה המספר של סי כי זה מה שיביא את ההתפלגות האחידה לכסות את הגרף של הצפיפות הנתונה
"""

def random_F(n):
    X=[]
    while len(X)<n:
        y=rn.rand(1)
        u=rn.rand(1)
        if (15*(1-y**2)**2/8)/(15*1/8)>=u:
            x=y
        else:
            continue
        X.append(float(x))
    return X
z=random_F(100000)                   #random normal numbers
x=np.linspace(0,1,10000)            #for trend line
y=1.875*(1-x**2)**2                 #for trend line
plt.plot(x,y,'--')                  #for trend line
plt.hist(z,bins=100, normed=1)      #F histogram
plt.show()
