# Yuval Lavie
# 305579872
# Financial Mathematics 2017


# Trajectory creation methods for Numerical Methods
# this file has all known trajectories and their E.M / Analytical solutions

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

# Analytic Solutions

# --------------------------------------------------------------------------
# Geometic Brownian Motion ( GBM )
# --------------------------------------------------------------------------
# dS(t) = S(t)*(rdt + sig*dWt)
# dS(t) = rS(t)dt + sigS(t)dWt

def GBMAnalytic(M,N,T,s0,r,sig):
    S = np.ones((M,N))*s0 # Skeleton
    dt = h = T/N
    dW = np.random.randn(M,N)*np.sqrt(h)
    for i in range(N-1):
        S[:,i+1] = S[:,i]*np.exp((r-(sig**2)/2)*dt + sig*dW[:,i])
    return S
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 2 Dimensional Geometic Brownian Motion ( GBM )
# --------------------------------------------------------------------------
# dS1(t) = S1(t)*(r1dt + sig1dWt)
# dS2(t) = S2(t)*(r2dt + sig2dWt)

def TwoDimGBM(M,N,T,s10,r1,sig1,s20,r2,sig2,rho):
    
    dt = h = T/N
    
    # Skeletons
    S1 = np.ones((M,N))*s10
    S2 = np.ones((M,N))*s20 
    
    # Cholesky's Decomposition
    Z1 = np.random.randn(M,N)*np.sqrt(h) # Z1 ~ N(0,h)
    Z2 = np.random.randn(M,N)*np.sqrt(h) # Z2 ~ N(0,h)

    dW1 = Z1;
    dW2 = rho*Z1 + np.sqrt(1-rho**2)*Z2


    for i in range(N-1):
        S1[:,i+1] = S1[:,i]*np.exp((r1-(sig1**2)/2)*dt + sig1*dW1[:,i])
        S2[:,i+1] = S2[:,i]*np.exp((r2-(sig2**2)/2)*dt + sig2*dW2[:,i])
    return S1,S2
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 3 Dimensional Geometic Brownian Motion ( GBM )
# --------------------------------------------------------------------------
# dS1(t) = S1(t)*(r1dt + sig1dWt)
# dS2(t) = S2(t)*(r2dt + sig2dWt)
# dS3(t) = S3(t)*(r3dt + sig3dWt)

def ThreeDimGBM(M,N,T,S0,R,SIG,corrMat):    
    h = T/N
    chol = np.linalg.cholesky(corrMat)

    # Skeletons
    S1 = np.ones((M,N))*S0[0]
    S2 = np.ones((M,N))*S0[1]
    S3 = np.ones((M,N))*S0[2]
     
    # Cholesky's Decomposition
    Z1 = np.random.randn(M,N)*np.sqrt(h) # Z1 ~ N(0,h)
    Z2 = np.random.randn(M,N)*np.sqrt(h) # Z2 ~ N(0,h)
    Z3 = np.random.randn(M,N)*np.sqrt(h) # Z3 ~ N(0,h)

    dW1 = chol[0,0]*Z1;
    dW2 = chol[1,0]*Z1 + chol[1][1]*Z2
    dW3 = chol[2,0]*Z1 + chol[2][1]*Z2 + chol[2][2]*Z3


    for i in range(N-1):
        S1[:,i+1] = S1[:,i]*np.exp((R[0]-(SIG[0]**2)/2)*dt + SIG[0]*dW1[:,i])
        S2[:,i+1] = S2[:,i]*np.exp((R[1]-(SIG[1]**2)/2)*dt + SIG[1]*dW2[:,i])
        S3[:,i+1] = S3[:,i]*np.exp((R[2]-(SIG[2]**2)/2)*dt + SIG[2]*dW3[:,i])
    return S1,S2,S3
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Cox-Ingersoll-Ross Model
# --------------------------------------------------------------------------
# dV(t) = a*[c - V(t)]*dt + b*sqrt(V)*dWt

def CIR(M,N,T,a,c,b,r0):
    A = np.zeros((M,N))
    R = np.ones((M,N)) * r0
    h = T/N
    dW = np.random.randn(M,N)*np.sqrt(h)
    # Euler's Scheme Approximation
    for i in range(N-1):
        A[:,i+1] = A[:,i] + b*(np.sqrt(np.abs(R[:,i])))*np.exp(a*i*h)*dW[:,i]
        R[:,i+1] = c + (r0 - c)*np.exp(-a*(i+1)*h)+np.exp(-a*(i+1)*h)*A[:,i+1]
    return R
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Heston Model
# --------------------------------------------------------------------------
# dS(t) = r*S(t)dt + sqrt(V(t))*S(t)dW1t
# dV(t) = a*[c - V(t)]*dt + b*sqrt(V)*dW2t

def Heston(M,N,T,gbmS0,gbmMu,CIRa,CIRc,CIRb,CIRv0,rho):
    # Skeletons
    S = np.ones((M,N))*gbmS0
    A = np.zeros((M,N))
    V = np.ones((M,N)) * CIRv0
    
    # dt
    h = T/N
    
    # Cholesky's Decomposition
    Z1 = np.random.randn(M,N)*np.sqrt(h) # Z1 ~ N(0,h)
    Z2 = np.random.randn(M,N)*np.sqrt(h) # Z2 ~ N(0,h)

    dW1 = Z1;
    dW2 = rho*Z1 + np.sqrt(1-rho**2)*Z2
    
    # Generate Trajectories
    for i in range(N-1):
        # GBM
        S[:,i+1] = S[:,i] * np.exp((gbmMu - (V[:,i] / 2))*h + np.sqrt(np.abs(V[:,i]))*dW1[:,i] )
        
        # CIR
        A[:,i+1] = A[:,i] + CIRb*(np.sqrt(np.abs(V[:,i])))*np.exp(CIRa*i*h)*dW2[:,i]
        V[:,i+1] = CIRc + (CIRv0 - CIRc)*np.exp(-CIRa*(i+1)*h)+np.exp(-CIRa*(i+1)*h)*A[:,i+1]
    return S,V
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Two Dimensional Heston Model
# --------------------------------------------------------------------------
# dS(t) = r(t)*S(t)dt + sqrt(V(t))*S(t)dW1t
# dV(t) = a*[b - V(t)]*dt + u*sqrt(V)*dW2t
# dR(t) = z*[c - R(t)]*dt + y*dW3t

def TwoDimHeston(M,N,T,gbmS0,gbmMu,CIRa,CIRc,CIRb,CIRv0,VASa,VASc,VASb,VASr0,rho):
    h = T/N
    # Skeletons
            # GBM S(t)
    # ------------------------------
    S = np.ones((M,N))*gbmS0
    
            # CIR V(t)
    # ------------------------------
    A = np.zeros((M,N))
    V = np.ones((M,N)) * CIRv0
    # ------------------------------
    
            # Vasicek R(t)
    # ------------------------------
    B = np.zeros((M,N))
    R = np.ones((M,N)) * VASr0
    # ------------------------------
    
    # ------------------------------
            # Cholesky
    # ------------------------------

    # Cholesky's Decomposition
    Z1 = np.random.randn(M,N)*np.sqrt(h) # Z1 ~ N(0,h)
    Z2 = np.random.randn(M,N)*np.sqrt(h) # Z2 ~ N(0,h)
    Z3 = np.random.randn(M,N)*np.sqrt(h) # Z3 ~ N(0,h)
    
    dW1 = Z1
    dW2 = Z2;
    dW3 = rho*Z2 + np.sqrt(1-rho**2)*Z3
 
    # Generate Trajectories
    for i in range(N-1):
        # GBM
        S[:,i+1] = S[:,i] * np.exp((gbmMu - (V[:,i] / 2))*h + np.sqrt(np.abs(V[:,i]))*dW1[:,i] )
        
        # CIR
        A[:,i+1] = A[:,i] + CIRb*(np.sqrt(np.abs(V[:,i])))*np.exp(CIRa*i*h)*dW2[:,i]
        V[:,i+1] = CIRc + (CIRv0 - CIRc)*np.exp(-CIRa*(i+1)*h)+np.exp(-CIRa*(i+1)*h)*A[:,i+1]
        
        # Vasicek
        B[:,i+1] = B[:,i] + VASb * np.exp(VASa * i * h)*dW3[:,i]
        R[:,i+1] = VASc + (VASr0 - VASc)*np.exp(-VASa* (i+1) *h) + np.exp(-VASa* (i+1) *h) * B[:,i+1]
    return S,V,R
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Brownian Bridge
# Takes W(0) and W(T) and returns brownian trajectories between them.
# --------------------------------------------------------------------------

# Takes z0 the probably that A occured.
def BrownianBridge(M,N,T,S0,r,sig,z0):
    
    # Skeleton
    S = np.ones((M,N))*S0
    
    # --------------------------------------------- #
                    # Calculate P(A)
    # --------------------------------------------- #
    
    # Calculate P(A)
    # Here A =: {S(T) < K}
    
    # P(S(T) < K) = P( Z < (ln(K/S0)-(r-(sig**2)/2)*T ) / sig * np.sqrt(T))
    z0=(np.log(K/S0)-(r-sig**2/2)*T)/(sig*np.sqrt(T))
    
    # Fn(Z < t) = cdf normal t
    P = ss.norm.cdf(z0)

    # --------------------------------------------- #
                    # Calculate E(X|A)
        # These are trajectories in the area of A
    # --------------------------------------------- #
    
        # We want to sample S(T) <=> sample Z < z0
                # Inverse Transform Method]
            
    U = np.random.rand(M); # U ~ U(0,1) , M samples one for each S(T)
    WT = ss.norm.ppf( U * P); # Phi^-1 (u * Phi(Z0)) = W ~ Z | Z < z0
    
    
    # --------------------------------------------- #
                    # Brownian Bridge
            # Fill in the middle of S from S0 to S(T)
    # --------------------------------------------- #
    
    # Skeleton
    h = T/N
    W = np.zeros((M,N-1))

    for i in range(1,N-1):
# build column i of Z ? using previous column and last column
        t = (i-1)*T/N
        a = (i-2)*T/N;
        b = T;
        sigma = np.sqrt( (t-a)*(b-t)/(b-a) );
        mu = W[:,i-1] + (t-a)/(b-a)*(WT-W[:,i-1]) ;
        W[:,i] = mu + sigma*np.random.randn(M);
        S[:,i]=S[:,i-1]*np.exp((r-sigma**2/2)*h+sig*(W[:,i]-W[:,i-1]));

    # Generate S(T)
    S[:,N-1] = S[:,N-2]*np.exp((r-(sig**2)/2)*h + sig*(WT-W[:,i]))
        
    return S
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Cox-Ingersoll-Ross Model with Jumps
# --------------------------------------------------------------------------
# dR(t) = a(b-R(t))dt + s(sqrt[R(t)])dWt +dJ(t)
# J(t) = Sum from k to N(t) of Y(k)
# where N(t) ~ Possion(lambda) number of jumps
# Y(k) ~ I.I.D (sometimes lognormal)

# --------------------------------------------------------------------------
# Cox-Ingersoll-Ross Model with Jumps
# --------------------------------------------------------------------------
# dR(t) = a(b-R(t))dt + s(sqrt[R(t)])dWt +dJ(t)
# J(t) = Sum from k to N(t) of Y(k)
# where N(t) ~ Possion(lambda) number of jumps
# Y(k) ~ I.I.D (sometimes lognormal)

def CIRJumps(T,observations,a,b,s,r0,jump_Lambda,jump_S,jump_Mu):
    dt = T/observations
    # Figure out how many jumps will occur
    t = np.linspace(0,T,num=observations) # Regular times with dt increments
    jump_Num = np.random.poisson(lam=jump_Lambda*T)
    #print(jump_Num)

    # Now we gotta add jumping times.
    jump_Times = np.random.uniform(0,T,size=jump_Num) # We start by adding jumps to the trajectory

    # Add Jump times to original times
    t = np.append(t,jump_Times) # Append regular times with jump times
    index = np.argsort(t) # Get's each time's index (Including the newly added jumps)
    t = t[index] # Sorts the array by the indices found above

    # Get jump indices in times
    jumps = np.append(np.zeros(observations),np.ones(jump_Num)) # This is used to specify where a jump occured
    jumps = jumps[index] # Arranges the jumps (0/1 yes/no) in the same order times is now ordered.

    # No jump -> Regular Trajectory
    R = np.ones(observations+jump_Num) * r0 # Skeleton with R[0] = r0
    for i in range(len(R)-1):
        #if(jumps[i] == 0):
            dt = t[i+1] - t[i]
            dW = np.random.randn()*np.sqrt(dt)
            R[i+1] = R[i] + a*(b-R[i])*dt + s*np.sqrt(np.abs(R[i]))*dW
        #else:
            if(jumps[i] == 1): # Jump Included
                Y = (np.sqrt(jump_S)*np.random.randn()) + jump_Mu # Y ~ N(mu,s^2)
                R[i+1] = R[i+1] + a*(Y-1) # a or jump_mu ?
    return R

# --------------------------------------------------------------------------


# --------------------------------------------------------------------------

# Euler - Maruyama Discretization

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Geometic Brownian Motion ( GBM )
# --------------------------------------------------------------------------
# dS(t) = S(t)*(rdt + sigdWt)
# dS(t) = rS(t)dt + sigS(t)dWt

def GBMEuler(M,N,T,s0,r,sig):
    S = np.ones((M,N))*s0 # Skeleton
    dt = h = T/N
    dW = np.random.randn(M,N)*np.sqrt(h)
    for i in range(N-1):
        S[:,i+1] = S[:,i] + (S[:,i] * r * h) + S[:,i]*sig*dW[:,i]
    return S
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Vasicek Model
# --------------------------------------------------------------------------
# dV(t) = a*[c - V(t)]*dt + b*dWt

def VasicekEuler(M,N,T,a,c,b,r0):
    R = np.ones((M,N)) * r0
    h = T/N
    dW = np.random.randn(M,N)*np.sqrt(h)
    # Euler's Scheme Approximation
    for i in range(N-1):
        R[:,i+1] = R[:,i] + a*(c - R[:,i])*h + b*dW[:,i]
    return R
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Cox-Ingersoll-Ross Model
# --------------------------------------------------------------------------
# dV(t) = a*[c - V(t)]*dt + b*sqrt(V)*dWt

def CIREuler(M,N,T,a,theta,b,v0):
    V = np.ones((M,N)) * v0
    h = T/N
    dW = np.random.randn(M,N)*np.sqrt(h)
    # Euler's Scheme Approximation
    for i in range(N-1):
        V[:,i+1] = V[:,i] + a*(theta - V[:,i])*h + b*np.sqrt(np.abs(V[:,i]))*dW[:,i]
    return V
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Heston Model
# --------------------------------------------------------------------------
# dS(t) = r*S(t)dt + sqrt(V(t))*S(t)dWt
# dV(t) = a*[c - V(t)]*dt + b*sqrt(V)*dWt


def HestonEuler(M,N,T,gbmS0,gbmMu,CIRa,CIRc,CIRb,CIRv0,rho):
    # Skeletons
    S = np.ones((M,N))*gbmS0
    V = np.ones((M,N)) * CIRv0
    
    # dt
    h = T/N
    
    # Cholesky's Decomposition
    Z1 = np.random.randn(M,N)*np.sqrt(h) # Z1 ~ N(0,h)
    Z2 = np.random.randn(M,N)*np.sqrt(h) # Z2 ~ N(0,h)

    dW1 = Z1;
    dW2 = rho*Z1 + np.sqrt(1-rho**2)*Z2
    
    # Generate Trajectories
    for i in range(N-1):
        # GBM
        S[:,i+1] = S[:,i] + (S[:,i] * gbmMu * h) + S[:,i]*(np.sqrt(np.abs(V[:,i])))*dW1[:,i]
        
        # CIR
        V[:,i+1] = V[:,i] + CIRa*(CIRc - V[:,i])*h + CIRb*np.sqrt(np.abs(V[:,i]))*dW2[:,i]
    return S,V
# --------------------------------------------------------------------------