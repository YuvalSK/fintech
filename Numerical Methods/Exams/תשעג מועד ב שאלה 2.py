"""
2א
"""
import numpy as np
from numpy import random as rn
from scipy.optimize import fsolve

S0=1
T=1
N=100
h=T/N
M=10000
s=0.5
r=0.1
dw=np.sqrt(h)*rn.randn(M,N)
S=S0*np.ones((M,N+1))
B=1.7
for i in range(0,N):
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*dw[:,i])
A=np.exp(-r*T)*((np.max(S,1)<=B)*(S[:,-1]>=S0)*S[:,-1]+(np.max(S,1)>B)+(S[:,-1]<S0)-(np.max(S,1)>B)*(S[:,-1]<S0))
V=[np.mean(A),np.std(A)/np.sqrt(M)]
print("V(",B,")=", V)

"""
2ב
"""
S=S0*np.ones((M,N+1))
B=2
for i in range(0,N):
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*dw[:,i])
A=np.exp(-r*T)*((np.max(S,1)<=B)*(S[:,-1]>=S0)*S[:,-1]+(np.max(S,1)>B)+(S[:,-1]<S0)-(np.max(S,1)>B)*(S[:,-1]<S0))
V=[np.mean(A),np.std(A)/np.sqrt(M)]
print("V(",B,")=", V)

"""
2ג
"""
S=S0*np.ones((M,N+1))
B=2.2
for i in range(0,N):
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*dw[:,i])
A=np.exp(-r*T)*((np.max(S,1)<=B)*(S[:,-1]>=S0)*S[:,-1]+(np.max(S,1)>B)+(S[:,-1]<S0)-(np.max(S,1)>B)*(S[:,-1]<S0))
V=[np.mean(A),np.std(A)/np.sqrt(M)]
print("V(",B,")=", V)

"""
לא ממש עובד עם ניוטון ראבסון.. אפשר לצייר גרף ולראות מתי פוגע ב1.
def F(t):
    S0=1
    T=1
    N=100
    h=T/N
    M=10000
    s=0.5
    r=0.1
    dw=np.sqrt(h)*rn.randn(M,N)
    S=S0*np.ones((M,N+1))
    B=t
    for i in range(0,N):
        S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*dw[:,i])
    A=np.exp(-r*T)*((np.max(S,1)<=B)*(S[:,-1]>=S0)*S[:,-1]+(np.max(S,1)>B)+(S[:,-1]<S0)-(np.max(S,1)>B)*(S[:,-1]<S0))
    return np.mean(A)-1
print(fsolve(F,1.8))
"""
