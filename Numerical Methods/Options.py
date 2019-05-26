# Yuval Lavie
# 305579872
# Financial Mathematics 2017

# Option Pricing methods for Numerical Methods

import numpy as np
import scipy.stats as ss

# --------------------------------------------------------------------------------
                                # Black & Scholes
# --------------------------------------------------------------------------------
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))

def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

def blsprice(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
        return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))
		

		
# --------------------------------------------------------------------------------
                                # Vanilla Options
# --------------------------------------------------------------------------------

def vCall(K,r,T,S):
    return np.exp(-r*T)*(S[:,len(S[0]) - 1]>K)*(S[:,len(S[0]) - 1] - K)

def vPut(K,r,T,S):
    return np.exp(-r*T)*(S[:,len(S[0]) - 1]<K)*(K - S[:,len(S[0]) - 1])

# --------------------------------------------------------------------------------
                                # Exotic Options - OUT
# --------------------------------------------------------------------------------

def DownAndOutPut(B,K,r,T,S):
    return np.exp(-r*T)*(np.min(S,1)>B)*(S[:,len(S[0]) - 1]<K)*(K-S[:,len(S[0]) - 1])

def DownAndOutCall(B,K,r,T,S):
    return np.exp(-r*T)*(np.min(S,1)>B)*(S[:,len(S[0]) - 1]>K)*(S[:,len(S[0]) - 1] - K)

def UpAndOutOptionPut(B,K,r,T,S):
    return np.exp(-r*T)*(np.max(S,1)<B)*(S[:,len(S[0]) - 1]<K)*(K - S[:,len(S[0]) - 1]) 

def UpAndOutOptionCall(B,K,r,T,S):
    return np.exp(-r*T)*(np.max(S,1)<B)*(S[:,len(S[0]) - 1]>K)*(S[:,len(S[0]) - 1] - K)

# --------------------------------------------------------------------------------
                                # Exotic Options - IN
# --------------------------------------------------------------------------------
  
def DownAndInPut(B,K,r,T,S):
    return np.exp(-r*T)*(np.min(S,1)<B)*(S[:,len(S[0]) - 1]<K)*(K-S[:,len(S[0]) - 1])

def DownAndInCall(B,K,r,T,S):
    return np.exp(-r*T)*(np.min(S,1)<B)*(S[:,len(S[0]) - 1]>K)*(S[:,len(S[0]) - 1] - K)

def UpAndInPut(B,K,r,T,S):
    return np.exp(-r*T)*(np.max(S,1)>B)*(S[:,len(S[0]) - 1]<K)*(K-S[:,len(S[0]) - 1])

def UpAndInCall(B,K,r,T,S):
    return np.exp(-r*T)*(np.max(S,1)>B)*(S[:,len(S[0]) - 1]>K)*(S[:,len(S[0]) - 1] - K)
	
# --------------------------------------------------------------------------------
                                # Exotic Options - Asian
# --------------------------------------------------------------------------------

def vCallAsn(K,r,T,S):
    return np.exp(-r*T)*(np.mean(S,1)>K)*(np.mean(S,1) - K)

def vPutAsn(K,r,T,S):
    return np.exp(-r*T)*(np.mean(S,1)<K)*(np.mean(S,1) - K)
	
# --------------------------------------------------------------------------------
                                # Asian - OUT
# --------------------------------------------------------------------------------

def DownAndOutPutAsn(B,K,r,T,S):
    return np.exp(-r*T)*(np.min(S,1)>B)*(np.mean(S,1)<K)*(K-np.mean(S,1))

def DownAndOutCallAsn(B,K,r,T,S):
    return np.exp(-r*T)*(np.min(S,1)>B)*(np.mean(S,1)>K)*(np.mean(S,1) - K)

def UpAndOutOptionPutAsn(B,K,r,T,S):
    return np.exp(-r*T)*(np.max(S,1)<B)*(np.mean(S,1)<K)*(K - np.mean(S,1)) 

def UpAndOutOptionCallAsn(B,K,r,T,S):
    return np.exp(-r*T)*(np.max(S,1)<B)*(np.mean(S,1)>K)*(np.mean(S,1) - K)

# --------------------------------------------------------------------------------
                                # Asian - IN
# --------------------------------------------------------------------------------
  
def DownAndInPutAsn(B,K,r,T,S):
    return np.exp(-r*T)*(np.min(S,1)<B)*(np.mean(S,1)<K)*(K-np.mean(S,1))

def DownAndInCallAsn(B,K,r,T,S):
    return np.exp(-r*T)*(np.min(S,1)<B)*(np.mean(S,1)>K)*(np.mean(S,1) - K)

def UpAndInPutAsn(B,K,r,T,S):
    return np.exp(-r*T)*(np.max(S,1)>B)*(np.mean(S,1)<K)*(K-np.mean(S,1))

def UpAndInCallAsn(B,K,r,T,S):
    return np.exp(-r*T)*(np.max(S,1)>B)*(np.mean(S,1)>K)*(np.mean(S,1) - K)
	
