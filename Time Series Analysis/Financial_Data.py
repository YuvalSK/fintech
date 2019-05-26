import numpy as np
import pandas_datareader.data as web
import datetime
from matplotlib import pyplot as plt
import statsmodels.graphics.tsaplots as tsaplots

start = datetime.datetime(2013, 12, 31)
end = datetime.datetime(2014, 12, 31)

stocks = ['SPY','GLD','OIL']
df = web.DataReader(stocks, 'google', start, end)
week_pct_change=weeks.pct_change()[1:]
week_pct_change.cov()
week_pct_change.corr()

a=plt.pcolor(week_pct_change.corr().values)
plt.colorbar()
a=plt.yticks(np.arange(0.5,len(df.Close.columns)+0.5),week_pct_change.columns)
a=plt.xticks(np.arange(0.5,len(df.Close.columns)+0.5),week_pct_change.columns,rotation=30)
plt.xlim((0,len(df.Close.columns)))
plt.ylim((0,len(df.Close.columns)))
plt.show()

days = df.Close.resample('W')
days_pct_change=days.pct_change()[1:]

for i in days_pct_change:
    tsaplots.plot_acf(days_pct_change[i] ,lags=10, title='correlogram - AR(' + i + ')')
plt.show()
