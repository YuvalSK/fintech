import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.tsa.arima_process import arma_generate_sample


'''
MA(1) process example
'''

N=1000
x=np.zeros(N)
z=np.random.randn(N)
beta1=0.5

for i in np.arange(1,N,1):
    x[i] = z[i] + beta1*z[i-1]
plt.plot(x);
plt.show()

'''
AR (2) process example
'''
tsaplots.plot_acf(x, lags=10, title='correlogram - AR(2)');
plt.show()


'''
ARMA(p,q) process example
'''


ar=[1, -0.9, 0.08]
ma=[1, 0.5, 0.9]
samps=arma_generate_sample(ar, ma, 5000)
plt.plot(samps);
tsaplots.plot_acf(samps, lags=40);
tsaplots.plot_pacf(samps, lags=40);
plt.show()