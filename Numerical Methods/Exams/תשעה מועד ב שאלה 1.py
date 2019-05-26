"""
תשע"ה מועד ב שאלה 1
"""
import numpy as np
from numpy import random as rn
import matplotlib.pyplot as plt 


b=[0.1,0.5,1,2,5]
M=50000
T=2
N=100
dw1=rn.randn(M,N)
X0=1
h=T/N
for x in b:
    X=X0*np.ones((M,N+1))
    for i in range(0,N):
        X[:,i+1]=X[:,i]+1.6*h*(1-X[:,i])+x*dw1[:,i]*np.sqrt(h)*np.sqrt(abs(X[:,i]))
    EX=np.mean(X[:,-1])
    VX=np.var(X[:,-1])
    error=np.std(X[:,-1])/np.sqrt(M)
    print("b=",x,"\n----------")
    print("EX2=",EX)
    print("error=",error)
    print("VX2=",VX)
    

