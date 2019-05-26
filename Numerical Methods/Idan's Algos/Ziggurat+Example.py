# ## Rejection method from 20 level Ziggurat:

#import packages
import numpy as np
from numpy import random as rn

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt



X=[1.261966719614, 1.839247314489, 2.498590174575, 3*np.pi]      #X axis
XX = [1,np.sin(X[0])/X[0],np.sin(X[1])/X[1],np.sin(X[2])/X[2],0] #Y axis


# In[374]:

n = 50000       # number of simulation
y = np.zeros(n)
i = 0          

while i<=(n-1):
    b=rn.randint(0,4);
    x=X[b]*(-1+2*rn.rand());
    if (rn.rand()*(XX[b]-XX[b+1])<((np.sin(x)/x)**2-XX[b+1])):
        y[i]=x;
        i+=1;

Axis = plt.hist(y,500)
Ax   = Axis[1][1:]
Ay   = Axis[0]/(n/54)

Axis = plt.hist(y,30)


Ax = Axis[1][1:]
Ay = Axis[0]

plt.plot(Ax,Ay/(50000/30))

x1 = np.linspace(-10,10,2001)
y1 = (np.sin(x1)/x1)**2

fig, ax = plt.subplots()
ax.plot(x1, y1, color='red')
ax.scatter(Ax,Ay)
fig.show()

