# price an american call option on the average price of two shares
#compare with multiD.py
import numpy as np
from numpy import random as rn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S10 = 15;
S20 = 30;
r = 0.05;
sigma1 = 0.1 ;
sigma2 = 0.08;
rho = -0.2;
T = 1;
K = 24;
M = 10**3;

ex_times=np.linspace(0,T,252);
N=np.size(ex_times)
Z = rn.randn(M,N-1,2);

S1=np.ones((M,N));
S2=np.ones((M,N));
S1[:,0]=S10*np.ones(M);
S2[:,0]=S20*np.ones(M);


for i in range(0,N-1):#generate paths of St1,St2
    h=ex_times[i+1]-ex_times[i];
    dW1=np.sqrt(h)*Z[:,i,0];
    dW2 =np.sqrt(h)*(rho * Z[:,i,0] +  np.sqrt(1-rho**2) * Z[:,i,1]) ; #by Cholesky's
    S1[:,i+1]=S1[:,i]*np.exp((r-sigma1**2/2)*h+sigma1*dW1);#analytic method
    S2[:,i+1]=S2[:,i]*np.exp((r-sigma2**2/2)*h+sigma2*dW2);

#Option pricing
Val=np.zeros((M,N));
A = (S1[:,-1]+S2[:,-1])/2 ;
Immediate_ex = (A-K)*(A>K);
Val[:,-1]=Immediate_ex; #Value at expiration date
for i in range(N-2,-1,-1):
    A = (S1[:,i]+S2[:,i])/2 ;
    Immediate_ex = (A-K)*(A>K);

    Design_Matrix=np.vstack((S1[:,i],S1[:,i]**2,S2[:,i],S2[:,i]**2,S1[:,i]*S2[:,i])); #basis functions
    Design_Matrix=Design_Matrix.T
    Y=Val[:,i+1]
    params=np.linalg.lstsq(Design_Matrix,Y)[0];
    #calculate continuation value for each path
    contval=params[0]*S1[:,i]+params[1]*S1[:,i]**2+params[2]*S2[:,i]+params[3]*S2[:,i]**2+params[4]*S1[:,i]*S2[:,i];

    Val[:,i]=(contval<Immediate_ex)*Immediate_ex+(contval>=Immediate_ex)*np.exp(-r*(ex_times[i+1]-ex_times[i]))*Val[:,i+1];


MEAN = np.mean(Val[:,0]);
ERR = np.std(Val[:,0])/np.sqrt(M);

print(MEAN, ERR)