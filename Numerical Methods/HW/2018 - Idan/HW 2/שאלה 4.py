# coding: utf-8
# Exercise Set 2, Question 4
##EM on Eq. dX=X(1-X)dt + sigma*dW
## calculate the probability of |X|>10 as unique value
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn

#sigma between given values of 0.1 - 0.6
sigma=np.linspace(0.1, 0.6,(0.6-0.1)/0.02+1)
prob=np.zeros(len(sigma))

Tmax=10
N=200 #number of values each simulation
h=Tmax/N #(1) division to N equall parts
t=np.linspace(0, Tmax,Tmax/h+1)

M=1000 #number of simulations
Z=rn.randn(M,N)

above_10 = []
for k in range(0,len(sigma)-1):
    # for each sigma, calculate  a random Brownian motion
    X=(np.zeros((M,N+1)))
    # begining value
    X[:,0]=X[:,0]+0.5
    #save all the sigma's options
    sig=sigma[k]

    count = 0
    for j in range(0,M):
        #for each simulation - total M


        for i in range(0,N):
            #how many times for each simulation
            #based on EM find higher values than 10 and replace them with 10
            if (np.abs(X[j,i])<10):

                X[j,i+1]=X[j,i]+h*X[j,i]*(1-X[j,i])+sig*np.sqrt(h)*Z[j,i];
            else:
                count+=1
                X[j,i+1]=10;


    print('finished: {0} out of {1} sigma'.format(k,(len(sigma))))

    #given sigma, save the average probability that the value is 10
    prob[k]=np.sum(X[:,N]==10)/M;
    
print('MC for the odds of the values to jump above 10 is:', float(count/(M*N)))

#calculate only for the last column
plt.plot(sigma[:-1],prob[:-1])
#We expect the probability of divergence to increase with sigma - 
#higher fluctuations gives more chance of divergence. 

#X=np.concatenate(X, axis=0 )
plt.show()

