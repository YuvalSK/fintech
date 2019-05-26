
# coding: utf-8

# In[1]:

from cvxopt import matrix, solvers
import numpy as np


# In[2]:

#markovitz function for finding the optimal with givin cov matrix
#help(solvers.qp)

def markowitz(mu):

    cov_matrix=np.array([[1.0,0.5,0.0],[0.5,1.5,-0.5],[0.0,-0.5,2]],dtype=np.float64)
    Constrains_left=np.array([[1,2,3],[1,1,1]],dtype=np.float64)
    Constrains_right=np.array([[mu,1]],dtype=np.float64).T
    P=matrix(cov_matrix) #From numpy to CVXOPT
    A=matrix(Constrains_left) #From numpy to CVXOPT
    b=matrix(Constrains_right) #From numpy to CVXOPT
    q=matrix([0.0,0.0,0.0]) 
    #using solver for optimization
    solution=solvers.qp(P,q,A=A,b=b)
    return np.array(solution['x']).T[0]


# In[3]:

#markovitz when shourt is forhibited
def markowitz_no_short(mu):

    cov_matrix=np.array([[1.0,0.5,0.0],[0.5,1.5,-0.5],[0.0,-0.5,2]],dtype=np.float64)
    Constrains_left=np.array([[1,2,3],[1,1,1]],dtype=np.float64)
    Constrains_right=np.array([[mu,1]],dtype=np.float64).T
    
    no_short_left=-np.eye(3,dtype=np.float64)
    no_short_right=np.zeros(3,dtype=np.float64)
    
    P=matrix(cov_matrix) #From numpy to CVXOPT
    A=matrix(Constrains_left) #From numpy to CVXOPT
    b=matrix(Constrains_right) #From numpy to CVXOPT
    q=matrix([0.0,0.0,0.0]) 
    #G=matrix([[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]])
    G=matrix(no_short_left)
    h=matrix(no_short_right)
    
    solution=solvers.qp(P,q,A=A,b=b,G=G,h=h)
    
    if solution['status']=='optimal':
        return np.array(solution['x']).T[0]
    else:
        #print("Error No Solution Found")
        return np.nan


# In[4]:

#short allowed:
solvers.options['show_progress']=False
print("Short allowed:")
print(markowitz(7))
print(markowitz(2))
print(markowitz(3))


# In[7]:

#Short not allowed:
solvers.options['show_progress']=False
print("Short not allowed:")
print(np.round(markowitz_no_short(3),decimals=3))


# In[ ]:



