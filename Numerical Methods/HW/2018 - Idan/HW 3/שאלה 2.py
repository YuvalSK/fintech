"""
Q2A
"""
import numpy as np
from numpy import random as rn

r=0.1
s=0.4  #סיגמא
S0=1
k=1.1
T=60/252
M=50000
n=120 #twice a day(= 60)
S=S0*np.ones((M,n+1))
dw=rn.randn(M,n)
h=T/n

for i in range(0,n):
    #GBM analitical - for changing h (time periods = twice a day)
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])

# max for each simulation
mm=np.max(S,1)-k
#today's price
nn=(mm>0)*mm*np.exp(-r*T)
#MontaCarlo E and V
V=[np.mean(nn),np.std(nn)/np.sqrt(M)]
print("א. Value=", V)

"""
Q2B
"""
r=0.1
s=0.4  #סיגמא
S0=1
k=1.1
T=60/252
M=50000
n=600
S=S0*np.ones((M,n+1))
dw=rn.randn(M,n)
h=T/n
for i in range(0,n):
    # GBM analitical - for changing h (time periods = twice a day)
    S[:,i+1]=S[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])

# max for each simulation
mm=np.max(S,1)-k
#today's price
nn=(mm>0)*mm*np.exp(-r*T)
#MontaCarlo E and V
V=[np.mean(nn),np.std(nn)/np.sqrt(M)]

print("ב. Value=", V)

"""
Q2C
"""
r=0.1
s=0.4  #סיגמא
S01=1
epsilon=0.00005
S02=1+epsilon
k=1.1
T=60/252
M=50000
n=600
S1=S01*np.ones((M,n+1))
S2=S02*np.ones((M,n+1))
#dw=rn.randn(M,n)
h=T/n
for i in range(0,n):
    #mixture models, calculate twice to check the impact on the result = delta
    S1[:,i+1]=S1[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])
    S2[:,i+1]=S2[:,i]*np.exp((r-s**2/2)*h+s*np.sqrt(h)*dw[:,i])

# max for each simulation
mm1=np.max(S1,1)-k
mm2=np.max(S2,1)-k
#today's price
nn1=(mm1>0)*mm1*np.exp(-r*T)
nn2=(mm2>0)*mm2*np.exp(-r*T)

delta=(nn2-nn1)/epsilon

print("ג. delta=", np.mean(delta),";",np.std(delta)/np.sqrt(M))

"""
Q2D
"""
r=0.1
s1=0.4  #סיגמא
S01=1
epsilon=0.00005
s2=0.4+epsilon
k=1.1
T=60/252
M=50000
n=600
S1=S01*np.ones((M,n+1))
S2=S01*np.ones((M,n+1))
#dw=rn.randn(M,n)
h=T/n
for i in range(0,n):
    #mixture models, calculate twice to check the impact on the result = vega

    S1[:,i+1]=S1[:,i]*np.exp((r-s1**2/2)*h+s1*np.sqrt(h)*dw[:,i])
    S2[:,i+1]=S2[:,i]*np.exp((r-s2**2/2)*h+s2*np.sqrt(h)*dw[:,i])

# max for each simulation
mm1=np.max(S1,1)-k
mm2=np.max(S2,1)-k
#today's price
nn1=(mm1>0)*mm1*np.exp(-r*T)
nn2=(mm2>0)*mm2*np.exp(-r*T)
vega=(nn2-nn1)/epsilon

print("ד. vega=", np.mean(vega),";",np.std(vega)/np.sqrt(M))


