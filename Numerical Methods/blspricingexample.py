import numpy as np

import matplotlib #redundant with Jupyter
matplotlib.use("Agg")
import matplotlib.pyplot as plt #will not be in the exam

from bls import blsprice


S=112.6;#price
K=100;#strike
r=0.02;#rate
sigma=np.linspace(0.01,1,101);
T=2/12;#time

C=blsprice('C',S,K,r,sigma,T)
P=blsprice('P',S,K,r,sigma,T)
##
#[Call, Put] = blsprice(Price, Strike, Rate, Time, Volatility, Yield)


#specific for pythonanywhere - not for the exam
fig=plt.figure()
ax = fig.add_subplot(111)
ax.plot(sigma,C)
fig.savefig('Call.png')
ax.cla()
ax.plot(sigma,P);
fig.savefig('Put.png')
