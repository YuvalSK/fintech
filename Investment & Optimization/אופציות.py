import numpy as np
from math import exp,sqrt
import matplotlib.pyplot as plt

'''
#First Question
'''

#first sub
def put(s0,k,sigma,T,r,M):

#function to calculate put price using gbm module


    Z=np.random.randn(M)

    #GBM Formula
    S_T=s0*np.exp((r-sigma**2)*T+sigma*sqrt(T)*Z)
    temp=0

    #loop to sum the values
    for i in S_T:
        temp+=max(k-i,0)
    put_price=temp*exp(-r*T)/M

    return put_price


print("the data entered, reflects a price of - ", put(100,103,0.2,0.5,0.03,10000))

#sec sub

sigma =[t / 1000 for t in range(100,500,4) ]
put_price=[put(100,103,x,0.5,0.03,1000) for x in sigma]
plt.plot(sigma, put_price,'y')
plt.title('sigma simulation - put price')
plt.show()

#third sub
Z=np.random.randn(1000)

sigma =[t / 1000 for t in range(100,504,4) ]
put_price=[put(100,103,x,0.5,0.03,1000) for x in sigma]
plt.plot(sigma, put_price,'y')
plt.title('same values - more volatile at the high prices')
plt.show()


#fourth sub
Z=np.random.randn(1000)

s0 =[t / 1000 for t in range(90000,112000,2000) ]
put_price=[put(x,103,0.2,0.5,0.03,1000) for x in s0]
plt.plot(s0, put_price,'y')
plt.title('s0 simulation - put price - as the s0 rise, the price is getting lower....')
plt.show()


'''
#Sec Question
'''

def sec_put(s0,k,dt,sigma,T,r,M):
    Z=np.random.randn(M)
    S_T=s0*np.exp((r-sigma**2)*T+sigma*sqrt(T)*Z)
    s=0
    for i in S_T:
        s+=max(k-i,0)
    put_price=s*exp(-r*(T-dt))/M
    
    S_dt=s0*np.exp((r-sigma**2)*dt+sigma*sqrt(dt)*Z)
    y=0
    for i in S_dt:
        y+=max(k-i,0)
    put_price_dt=max(put_price*exp(-r*(dt)),y*exp(-r*(dt))/M)
    return put_price_dt

#without limiting the M, we will get very high number due to the recursion the depends on the numbers of calculation! (two expodentional values)
print("the price is - ", sec_put(100,103,0.25,0.2,0.5,0.03,1000))
