import numpy as np
from numpy import random as rn

def commonRV(flag,h,M):
    # option on average price of two assets obeying coupled GBM
    # want sensitivity of option price with respect to one underlying
    # using M simulations for each of f(x+h) and f(x-h)
    # will find derivative using f'(x) = ( f(x+h) - f(x-h) ) / (2h)
    # symmetrical which is more efficient

    # all option parameters
    S10 = 15;
    S20 = 30;
    r = 0.05;
    sigma1 = 0.1 ;
    sigma2 = 0.08;
    rho = 0.4;
    T = 1;
    K = 24;

    # "f(x+h)"
    Z = rn.randn(M,2);
    V2 = rho * Z[:,0] +  np.sqrt(1-rho**2) * Z[:,1] ;
    S1 = (S10 + h) * np.exp( (r-sigma1**2/2)*T + sigma1 * np.sqrt(T) * Z[:,0] );
    S2 = S20 * np.exp( (r-sigma2**2/2)*T + sigma2 * np.sqrt(T) * V2 );
    A = (S1+S2)/2 ;
    P1 = np.exp(-r*T)*(A-K)*(A>K);

    # "f(x-h)"
    if flag==1:#notice we draw new Z only if flag==1
        Z  = rn.randn(M,2);
        V2 = rho * Z[:,0] +  np.sqrt(1-rho**2) * Z[:,1] ;

    S1 = (S10 - h) * np.exp( (r-sigma1**2/2)*T + sigma1 * np.sqrt(T) * Z[:,0] );
    S2 = S20 * np.exp( (r-sigma2**2/2)*T + sigma2 * np.sqrt(T) * V2 );
    A = (S1+S2)/2 ;
    P2 = np.exp(-r*T)*(A-K)*(A>K);
    delta = (P1-P2)/(2*h) ;
    vals=[np.mean(delta),np.std(delta)/np.sqrt(M)]
    return vals

M=10**6
h = 10**-2

print(commonRV(0, h, M))
print(commonRV(1, h, M))