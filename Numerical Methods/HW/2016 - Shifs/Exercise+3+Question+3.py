# coding: utf-8
# Exercise Set 3, Question 3
# In[2]:
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn
#part a: Here is my program to compute the probability for a fixed value of K:

r=0.1
s=0.4
S0=1.0
K=1.1
T=60/252
N=600      
M=50000
h = T/N
Z = rn.randn(M,N)

S  = S0*np.ones((M,N+1))

# make prices
for i in range(0,N):
    S[:,i+1]  = S[:,i]*np.exp((r-s**2/2)*h + np.sqrt(h)*s*Z[:,i])

Smax  = np.max(S,1)
p = np.sum(Smax>K)/M
err = np.sqrt(p*(1-p)/M) 


# Can you explain the formula for the stochastic error?
# 
# Here is the same made to run for a range of values of K, with plots: 

# In[8]:

#part a: Here is my program to compute the probability for a fixed value of K:

r=0.1
s=0.4
S0=1.0
K=1.1
T=60/252
N=600      
M=50000
h = T/N
Z = rn.randn(M,N)

S  = S0*np.ones((M,N+1))

# make prices
for i in range(0,N):
    S[:,i+1]  = S[:,i]*np.exp((r-s**2/2)*h + np.sqrt(h)*s*Z[:,i])

Smax  = np.max(S,1)
K   = np.linspace(0.95, 1.6,(1.6-0.95)/0.002+1)
p   = np.zeros(len(K));
err = np.zeros(len(K));

for i in range(0,len(K)):
  p[i] = np.sum(Smax>K[i])/M
  err[i] = np.sqrt(p[i]*(1-p[i])/M)


# In[13]:

plt.plot(K,p)
plt.title("probability as a function of K")
plt.xlabel("K")
plt.ylabel("probability")


# In[14]:

plt.plot(K,err)
plt.title("stochastic error as a function of K")
plt.xlabel("K")
plt.ylabel("stochastic error")


# The probability is 1 if K is less than 1 and falls as K increases.
# 
# Part b: When K is large, most of the time the option return is zero. So in our simulation of the price 
# (using the program from question 2) most runs are wasted, as they just give zero. 
# For K=1.4 the probability of a positve return is less than 0.1, so 90% of the work is wasted. 
# If we use random variables with a N(mu,1) distribution with mu>0 we will spend more time 
# exploring the "interesting" region and have less wasted runs.
# 

# Part c: Here is a program that computes using N(mu,1) random numbers. 
# There are 2 changes - the random numbers are chosen differently, 
# and the return function has to be modified (the formula for this was proved in class). 
# NOTE: this year I did not discuss how mu should change with N. 
# But since in this program I keep N fixed at 600, we can ignore this issue. 

# In[20]:

r=0.1
s=0.4
S0=1.0
K=1.1
T=60/252
N=600      
M=50000
h = T/N

# make the random numbers with a N(mu,1) distribution
mu = 0.1
Z  = mu +rn.randn(M,N)

S  = S0*np.ones((M,N+1))

# make prices
for i in range(0,N):
    S[:,i+1]  = S[:,i]*np.exp((r-s**2/2)*h + np.sqrt(h)*s*Z[:,i])

Smax  = np.max(S,1);
ret = np.exp(-r*T)*(Smax-K)*(Smax>K);
# modify the return to take into account the new distribution
# formula was explained in class 
retmod = ret * np.exp(-mu*np.sum(Z,1)*np.exp(mu**2*N/2))
np.mean(retmod)
np.std(retmod)/np.sqrt(M)



# In[22]:

#Part d: Absolutely no reason not to use this method to compute the Delta!! 
#We will compare mu=0 and mu=0.07. Here is my program, I hope I got it all right:

r=0.1
s=0.4
S0=1.0
T=60/252
K=1.4
N=600       
M=50000      
h = T/N
eps = 0.0001    # for computing the delta

# make the random numbers with a N(mu,1) distribution
mu = 0.07
Z = mu+rn.randn(M,N);

# make two matrices of prices
Splus  = (S0+eps)*np.ones((M,N+1))
Sminus = (S0-eps)*np.ones((M,N+1))

# make prices
for i in range(0,N):
    Splus[:,i+1]  = Splus[:,i]*np.exp((r-s**2/2)*h + np.sqrt(h)*s*Z[:,i])
    Sminus[:,i+1]  = Sminus[:,i]*np.exp((r-s**2/2)*h + np.sqrt(h)*s*Z[:,i])


# make returns and modified returns
Splusmax  = np.max(Splus,1)
retplus   = np.exp(-r*T)*(Splusmax-K)*(Splusmax>K)
retmodplus = retplus*np.exp(-mu*np.sum(Z,1))*np.exp(mu**2*N/2) 

Sminusmax  = np.max(Sminus,1)
retminus   = np.exp(-r*T)*(Sminusmax-K)*(Sminusmax>K)
retmodminus = retminus *np.exp(-mu*np.sum(Z,1))*np.exp(mu**2*N/2) 

sens = (retmodplus - retmodminus)/(2*eps);
np.mean(sens)
np.std(sens)/np.sqrt(M)

#With mu=0.007 I got typical answers of 0.125 with stochastic errors of 0.001.
#With mu=0 I got answers in the range 0.122 to 0.130 with stochastic errors of 0.002. 
#Using importance sampling here has reduced the stochastic error by about 40% 
#(it looks like 50% but I have rounded generously). This was for a value of 
#K where "only" 90% of simulations were being wasted. For higher K the variance 
#reduction would have been even more pronounced. 

