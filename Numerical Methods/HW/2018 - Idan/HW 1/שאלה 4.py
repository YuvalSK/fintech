#!/usr/bin/env python3
"""
Q4A
"""
import numpy as np

def four_cube_expectation(n):
    X=[]
    for x in range(n):                                                  #Number of iterations
        u=np.random.randint(1,7,size=4)                                 #4 random Cube injections
        x=abs(np.sum(u)-14)
        X.append(x)
    return([np.mean(X),np.var(X,ddof=1)])                               #expectation and var
"""
Q4B
"""

def bigger_cube_number_five(n):
    X=[]
    for x in range(n):                                                  #Number of iterations
        u=np.random.randint(1,7,size=5)                                 #5 random Cube injections
        x=np.max(u)                                                     #maximum from five injections
        X.append(x)
    return([np.mean(X),np.var(X,ddof=1)])
"""
Q4C
"""

def counting_cube_six_out_of_ten(n):
    X=[]
    for x in range(n):                                                  #Number of iterations
        u=np.random.randint(1,7,size=10)                                #10 random Cube injections
        x=np.count_nonzero(u==6)                                        #count how many six out of ten injections
        X.append(x)
    return([np.mean(X),np.var(X,ddof=1)])
"""
Q4D
"""

def counting_most_common(n):
    X=[]
    for x in range(n):                                                  #Number of iterations
        u=np.random.randint(1,7,size=20)                                #20 random Cube injections
        x=np.max(np.bincount(u))                                        #count how many common number in X
        X.append(x)
    return([np.mean(X),np.var(X,ddof=1)])

print("Q4A. [E(X),S^2(X)]=",four_cube_expectation(100000))
print("Q4B. [E(X),S^2(X)]=",bigger_cube_number_five(100000))
print("Q4C. [E(X),S^2(X)]=",counting_cube_six_out_of_ten(100000))
print("Q4D. [E(X),S^2(X)]=",counting_most_common(100000))


