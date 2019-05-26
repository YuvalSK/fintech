import numpy as np
from numpy import random as rn
import matplotlib
import matplotlib.pyplot as plt

#setting up constant parameters
h=1/50
M=1000
T=1
r0=5
K=5

#(1) skulete adjustment
lamda = np.linspace(0.5,10,50)
#mean for the price
m_p=np.zeros(np.shape(lamda))
#varience for the price
var_p=np.zeros(np.shape(lamda))


for i in range(0, len(lamda)):
    # lamda is changing, so calculos for each lamda values and calculate (price,error) for each
    p=np.zeros(M)
    for x in range(0,M):
        #simulations for each given lamda

        #(2) - diffrent random poi values for jumps = N(t)~poi(lamda)
        N=rn.poisson(lam=lamda[i]*T)

        #(3) diffrent random uniform values = time
        tau=rn.uniform(0,T,N)

        #(4) - random normal values ~N(1, 0.1)
        y = rn.normal(1, 0.1, size=N)

        #(5) - building times periods (0-T)
        n=np.linspace(0,T,T/h)
        t=np.append(n, tau) #random time values for exam
        index=np.argsort(t) #sorts the time h and values 0-1
        t=t[index]
        #setting the jumps for each time above
        jump=np.append(np.zeros(int(T/h+1)), np.ones(N))
        jump=jump[index]

        #counter for the jumps
        k=-1

        #(6) Eq for the begining values for process
        # r for the Eq.
        r = r0 * np.ones(len(t) + 1)
        #normail dist for brownian process later
        z=rn.randn(len(t))
        # r(1), r(0)
        r[1]=r[0]+2*(5-r[0])*t[0]+0.2*np.sqrt(r[0])*np.sqrt(t[0])*z[0]

        for j in range(1,len(r.T)-1):
            #Eq. for the process
            r[j+1]=r[j]+2*(5-r[j])*(t[j]-t[j-1])+0.2*np.sqrt(r[j])*np.sqrt(t[j]-t[j-1])*z[j]

            #(7) - check the jumps and add when needed
            if jump[j]==1:
                k+=1
                r[j+1]=r[j+1]*y[k]

        #today's price
        pi=(r[-1]-K)*(r[-1]>K)*np.exp(-r[-1]*T)
        p[x]=pi


    #saving the mean and varience of the results for each iteration
    m_p[i]=np.mean(p)
    var_p[i]=np.std(p)/np.sqrt(M)

    # printing log
    if i % 10 == 0:
        print("{0} / {1} lamdas\nMC Vanila Call price: {2}, Error: {3}".format(i, len(lamda),np.mean(p),np.std(p)/np.sqrt(M)))

# visualize the results
plt.plot(lamda, m_p)
plt.xlabel("lamda_poi_dist")
plt.ylabel("Price_mean")
plt.show()
