#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats as ss

def random_F_Ziggurat(N):
    #X - axis #Y - axis
    A=np.array([[ 0, 0.3977/1.0992+0.4361 ], #Zero floor
    [1.0992, 0.4361],
    [1.6697, 0.1979]])   
    X=[]
    v=0.3977
    r=1.6697
    while len(X)<N:
        n=np.random.randint(0,3)                      #choose rectangle
        if n==2:
            u=np.random.rand(1)
            x=u*v/(2*np.exp(-0.5*r**2)/np.sqrt(2*np.pi))
            if x<=r:
                X.append(float(x))
            else:
                u=np.random.rand(1)
                z=ss.norm.ppf(u)
                while z<r:
                    u=np.random.rand(1)
                    z=ss.norm.ppf(u)
                X.append(float(z))             
        else:
            x=np.random.rand(1)*(A[n+1][0]-A[0][0])+A[0][0]  #random x in rectangle
            f=2*np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
            if f>A[n][1]:
                X.append(float(x))
            else:
                y=np.random.rand(1)*(A[n][1]-A[n+1][1])+A[n+1][1]
                if y<=f:
                    X.append(float(x))
                else:
                    continue
    return X
z=random_F_Ziggurat(100000)                   #random normal numbers
s=np.linspace(0,3,10000)            #for trend line
t=2*np.exp(-0.5*s**2)/np.sqrt(2*np.pi)                #for trend line
plt.plot(s,t,'--')                  #for trend line
plt.hist(z,bins=100, normed=1)      #F histogram
plt.show()
