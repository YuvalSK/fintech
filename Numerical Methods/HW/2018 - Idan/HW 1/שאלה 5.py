#!/usr/bin/env python3
"""
Q5
"""
import numpy as np

def n_ball_volume(n):
    N=1000000
    v=0
    for i in range(N+1):
        x=2*np.random.rand(n)-1
        if sum(x**2)<1:
            v+=1
        else:
            continue
    return v/N

n=10
print("Volume={0:.3f}".format((2**n)*n_ball_volume(n)),"\nstd= {0:.3f}".format((2**n)*np.sqrt((n_ball_volume(n)-n_ball_volume(n)**2)/1000000)))

