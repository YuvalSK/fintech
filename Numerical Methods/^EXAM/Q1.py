import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as ss

def Ziggurat(N):

    # choose rectangle Ziggurate
    A=np.array([[ 0, 0.3977/1.0992+0.4361],
    [1.0992, 0.4361],
    [1.6697, 0.1979]])

    X=[]
    v=0.3977
    r=1.6697

    while len(X)<N:

        #Random: A, B, C
        n=np.random.randint(0,3)
        if n==2:

            #if C is choosen take a random sample
            u=np.random.rand(1)

            #create X vals by the rule - X = U * (v /f(r))
            def f(r):
                2 * np.exp(-0.5 * r ** 2) / np.sqrt(2 * np.pi)

            Func = v/f(r)
            x=u*v/(Func)

            if x<=r:
                #if left to the X0 - we return as a vaild sample
                X.append(float(x))
            else
                #Random Z~N(0,1)
                u=np.random.rand(1)
                z=ss.norm.ppf(u)
                while z<r:
                    u=np.random.rand(1)
                    z=ss.norm.ppf(u)
                X.append(float(z))
        else:

            x=np.random.rand(1)*(A[n+1][0]-A[0][0])+A[0][0]
            # Inverse Function
            f=2*np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
            if f>A[n][1]:
                X.append(float(x))
            else:
                y=np.random.rand(1)*(A[n][1]-A[n+1][1])+A[n+1][1]
                if y<=f:
                    X.append(float(x))
                else:
                    continue
    return X

def main():

    #number of simulations
    M = 10**5
    #visuallization of the results
    s = np.linspace(0,3,10000)
    t = 2*np.exp(-0.5*s**2)/np.sqrt(2*np.pi)

    #Ziggurat by the defenition of the
    z = Ziggurat(M)

    #histogram
    plt.hist(z, bins=100, normed=1)
    plt.title('Ziggurate Random Sampling - Yuval')
    #function to compare the simulation results
    plt.plot(s,t,'--')

    plt.show()

main()