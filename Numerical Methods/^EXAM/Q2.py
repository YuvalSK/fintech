import numpy as np
from numpy import random as rn

import scipy.stats as ss

def DownAndInCall(H,K,r,T,S):
    return np.exp(-r*T)*(np.min(S,1)<H)*(S[:,len(S[0]) - 1]>K)*(S[:,len(S[0]) - 1] - K)

#parameters
T=1
n=100
h=T/n
a=1
b=0.5
r=0.5
mu=0.02
rho=0.1
S0=10
teta=0.2
v0=teta
k=10
M=1000
H=8
M=10000

dw1=rn.randn(M,n)
dw2=rn.randn(M,n)

S1=S0*np.ones((M,n+1))
v=v0*np.ones((M,n+1))
LR = np.ones((M,n+1))


for i in range(0,n):
    S1[:, i + 1] = S1[:, i] * np.exp((mu - v[:, i] / 2) * h + np.sqrt(np.abs(v[:, i]) * h) * dw1[:, i])
    v[:,i+1]=v[:,i]+a*(teta-v[:,i])*h+b*np.sqrt(h*np.abs(v[:,i]))*(rho*dw1[:,i]+np.sqrt(1-rho**2)*dw2[:,i])

    if i == T or i==T/2:
        # importance sampling
        #f(x)
        mm = DownAndInCall(H, k, v[i], i, S[i])
        #f(y)
        y = np.exp(-mu * T) * (S1[:, -1] > k) * (S1[:, -1] - k)

        # h*(y) = f(x)/f(y)
        LR =  mm / np.linalg.lstsq(np.vstack([mm, np.ones(len(mm))]).T, y)[0][0] * y

        # h*(y) = f(x)/f(y)
        nn = mm * LR

        V.append(np.mean(nn))
        err.append(np.std(nn) / np.sqrt(M))

print('the MC result is\n E(x) = {1}, stochastic error = {12}'.format(np.mean(V), np.mean(err)))


