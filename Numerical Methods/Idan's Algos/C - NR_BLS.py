import numpy as np
import scipy.stats as ss


#Black and Scholes
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))

def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))

def blsprice(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))


def bsput(sigma):
	S=112.6;
	K=100;
	r=0.02;
	T=2/12;
	return blsprice('P',S,K,r,sigma,T)

def newtstepBS(x1):
	P=5;#known from the market
	u=10**-5
	df=(bsput(x1+u)-bsput(x1))/u; #approximation for f'(x)

	return x1-(bsput(x1)-P)/df;#x1-f(x1)/f'(x1)




