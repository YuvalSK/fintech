"""
Q6
"""
import numpy as np
from numpy import random as rn

S0=1
T=5
h=0.005
N=int(T/h)
M=10000
s=0.1
r=0.15
S=S0*np.ones((M,N+1))
dw=np.sqrt(h)*rn.randn(M,N)

A=[]
for x in range(0,M):       #with bank investment
    for i in range(0,N):
        S[x,i+1]=S[x,i]*np.exp((r-0.5*s**2)*h+s*dw[x,i])
        #the investment simulated value
        if S[x,i+1]>=1.75:
            SS=S[x,i+1]*np.exp(0.05*(T-(i+1)*h))
            break
        else:
            SS=S[x,i+1]
    A.append(SS)

print("Profit with bank investment:",np.mean(A),"\nerror:",np.std(A)/np.sqrt(M))
B0=1
B=B0*np.ones((M,N+1))    #without bank investment
for i in range(0,N):
    B[:,i+1]=B[:,i]*np.exp((r-0.5*s**2)*h+s*dw[:,i])
print("Profit without bank investment:",np.mean(B[:,-1]),"\nerror:",np.std(B[:,-1])/np.sqrt(M))

#Answers (reproduced in several runs) gave the expected value of
#the "sell high" strategy to be about 1.83 and the expected value of the
#"hold 5 years" strategy to be about 2.12
#(the exact value for this, from question 3, is exp(0.75)=2.1170).
#BUT the standard deviation of the first strategy is only about 0.18 while for the second strategy
#it is 0.48. Thus, as usual in these things, the "cost" of a strategy
#with a high average return is high risk.
#It is interesting to look at the distributions of the returns from the 2 strategies,
#beyond just their means and variances, to decide what is relevant for an investment decision.
