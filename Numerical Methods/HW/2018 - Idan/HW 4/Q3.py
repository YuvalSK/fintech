import numpy as np
from numpy import random as rn

#barmuda call
#setting up parameters
S10=100
S20=100
K=100
T=3
M=100000
ex_times=np.linspace(0,T,12)
N=np.size(ex_times)

#(2) - M simulations for values
z=rn.randn(M,N-1,2)

#deferential parameters
rho=-0.5
r1=0.1
r2=0.1
s1=0.1
s2=0.2
S1=S10*np.ones((M,N))
S2=S10*np.ones((M,N))

#(3) - loop to find the values of the assets
for i in range(0,N-1):
    h=ex_times[i+1]-ex_times[i]
    #define the brownian motions
    dw1=np.sqrt(h)*z[:,i,0]
    dw2=np.sqrt(h)*(rho*z[:,i,0]+np.sqrt(1-rho**2)*z[:,i,1])

    #the equations for the asset as given
    S1[:,i+1]=S1[:,i]*np.exp((r1-s1**2/2)*h+s1*dw1)
    S2[:,i+1]=S2[:,i]*np.exp((r2-s2**2/2)*h+s2*dw2)

val=np.zeros((M,N))
#A = the value of call barmuda option max(s1,s2)
A=(S1[:,-1]>S2[:,-1])*S1[:,-1]+(S2[:,-1]>S1[:,-1])*S2[:,-1]
# the higher asset as call (s-k)
ex=(A-K)*(A>K)
val[:,-1]=ex

#loop to test all the values from the end to the top
for i in range(N-2,-1,-1):
    A=(S1[:,i]>S2[:,i])*S1[:,i]+(S2[:,i]>S1[:,i])*S2[:,i]
    ex=(A-K)*(A>K)
    Matrix=np.vstack((S1[:,i],S1[:,i]**2,S2[:,i],S2[:,i]**2,S1[:,i]*S2[:,i],ex))
    Matrix=Matrix.T
    Y=val[:,i+1]
    # (4) - linear regression - function for each simulation
    params=np.linalg.lstsq(Matrix,Y)[0]

    #(1) function to test regression
    contval=params[0]*S1[:,i]+params[1]*S1[:,i]**2+params[2]*S2[:,i]+params[3]*S2[:,i]**2+params[4]*S1[:,i]*S2[:,i]+params[5]*ex
    #(5) - save the last value in today's price
    val[:,i]=(contval<ex)*ex+(contval>=ex)*np.exp(-r1*(ex_times[i+1]-ex_times[i]))*val[:,i+1]
    print("steps left to finish {} ".format(i))

#we got a list with today's value options, the mean is the price!
mean = np.mean(val[:,0])
std = np.std(val[:,0])/np.sqrt(M)

print("The mean is: {0}\nstd: {1}".format(mean, std))