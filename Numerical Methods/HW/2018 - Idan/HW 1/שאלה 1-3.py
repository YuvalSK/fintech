import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rn
import matplotlib.mlab as mlab
import matplotlib
from scipy.optimize import fsolve

# (1) LSG - correlation between x and y for constant m, a,c
##    y =(ax + c) mod m
## The results indicats that there is a very low correlation and therefor strong Independence
## the more simulation the lower the corr gets

'''
m=2147483647
a=69621
c=0
x=np.random.randint(1,m-1,10000)
y=(a*x+c)%m
z=a*x+c

print('Correlation between x.y is: {0}\n given {1} simulations \n (Eq: y=(a*x+c)%m)'.format(np.corrcoef([x, y])[1,0], len(x)))
'''

# (2) Generate Random normal numbers from Uniform numbers X~U[0,1] using different methods

## (A) Eq. X = (index i=1-12)sigma i (u) - 6
def Random_Normal(n):
    X=[]
    D = 12 # numbers to simulate

    #Loop to generate random numbers u~U[0,1]
    ## Sums up the random numbers and preforms the given Eq.
    for i in range(n):
        u=rn.rand(D) # Uniform numbers
        x=np.sum(u)-6 #
        X.append(x)
    return X

## (B) Eq. BoxMuller
def Random_Normal_Box_muller(n):
    X=[]
    # Loop to generate random numbers u~U[0,1]
    ## preforms the given Eq.
    for i in range(int(n/2)):
        u=rn.rand(2) #generats U1,U2~U[0,1]
        # R = sqrt(-2ln(u2)), teta = 2*pai*u1)
        z1=np.sqrt(-2*np.log(u[1]))*np.sin(2*np.pi*u[0]) # Z1 = R * cos (teta)
        z2=np.sqrt(-2*np.log(u[1]))*np.cos(2*np.pi*u[0]) # Z2 = R * sin(teta)
        X.append(z1)
        X.append(z2)
    return X

## (C) Marsaglia's improvement to BoxMuller's
def Random_Normal_Margsalia_Imp(n):
    X=[]

    while len(X)<n:
        u=rn.rand(2) #generats U1,U2~U[0,1]
        v1=2*u[0]-1 #generats V1,V2~U[-1,1]
        v2=2*u[1]-1

        # if the random number is inside the circle, we save the numbers (normal), while rejecting them otherwise
        if (v1)**2+(v2)**2<1:
            r=(v1)**2+(v2)**2
            z1=v1*np.sqrt(-2*np.log(r)/r)  # Z1 = V1 * sqrt (-2 lnR/R)
            z2=v2*np.sqrt(-2*np.log(r)/r)  # Z2 = V2 * sqrt (-2 lnR/R)
            X.append(z1)
            X.append(z2)
        else:
            continue
    return X

'''
z= Random_Normal(100000)
#z=Random_Normal_Box_muller(100000)
#z=Random_Normal_Margsalia_Imp(100000)
x=np.linspace(-4,4,10000)             
plt.plot(x,mlab.normpdf(x,0,1),'--') 
plt.hist(z,bins=1000, normed=1)
plt.show()
'''

# (3) based on Eq. y=15/8 (1-X^2)^2 for 0<x<1 Generate a random numbers
## (A) Transpose Method
def random_F(n):
    X=[]
    for i in range(n):
        u=rn.rand(1) # X~U[0,1]
        #based on the random number, calculates the specific solution
        def F(t):
            S = 15*t/8-5*t**3/4+3*t**5/8-u
            return S
        x=fsolve(F,0.01)     # root of a function, 0.01 starting estimate
        X.append(float(x))
    return X

## (B) Rejection Sampling + Ziggurat - Integral estimation
def random_F_Ziggurat(N):
    A=np.array([[ 0, 1.875000000000000], 
    [0.243700966784892, 1.658900379046860],
    [0.326107066633865, 1.497408366341702],
    [0.384904660677814, 1.360585688083304],
    [0.432611507516508, 1.238851324495972],
    [0.473788396832940, 1.127696878966890],
    [0.510671893321394, 1.024570610932918],
    [0.544552878330669, 0.927860651912006],
    [0.576262880271060, 0.836472349192757],
    [0.606385282511440, 0.749623792562287],
    [0.635361459557960, 0.666736035119904],
    [0.663551251394990, 0.587369616267762],
    [0.691272577868383, 0.511185939619582],
    [0.718832774671270, 0.437923160843028],
    [0.746560490929985, 0.367381406775015],
    [0.774848504999009, 0.299414976939616],
    [0.804227184885832, 0.233931382828638],
    [0.835520550158683, 0.170900393898353],
    [0.870266300345544, 0.110385943864877],
    [0.912363601081001, 0.052663686548078],
    [1.000000000000000, 0]]) 

    X=[]
    while len(X)<N:
        n=np.random.randint(0,20)                   
        x=np.random.rand(1)*(A[n+1][0]-A[0][0])+A[0][0]  
        f=1.875*(1-x**2)**2
        if f>A[n][1]:
            X.append(float(x))
        else:
            y=np.random.rand(1)*(A[n][1]-A[n+1][1])+A[n+1][1]
            if y<=f:
                X.append(float(x))
            else:
                continue
    return X

'''
z=random_F(10000)
#z=random_F_Ziggurat(100000)
s=np.linspace(0,1,10000)            
t=1.875*(1-s**2)**2                 
plt.plot(s,t,'--')                  
plt.hist(z,bins=100, normed=1)      
plt.show()

'''
