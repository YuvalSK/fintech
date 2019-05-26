#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
import statsmodels.tsa.api as smt

def correlogram(data):
    plt.clf()
    ts = pd.Series(data)

    plt.figure()
    autocorrelation_plot(ts)
    plt.show()
def tsplot(y, lags=None, figsize=(10, 8)):

    plt.clf()
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    #pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    #smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(1.5) for ax in [acf_ax]] #pacf_ax]]
    sns.despine()
    plt.tight_layout()
    plt.show()

    #return ts_ax, acf_ax, pacf_ax
def autocorr(x, t=1):   # פונקציה לחישוב מקדם התאמה עצמי
    return float(np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[1,0])


newpath = r'output'

if not os.path.exists(newpath):
    os.makedirs(newpath)

try:

    with open("Data\\ggg.csv", mode="rt", encoding="utf8") as ggg, \
         open(os.path.join(newpath,'correlationday.csv'), mode="wt", encoding="utf8") as correlationday:
        ggg = csv.reader(ggg, delimiter=",")
        for row in ggg:
            #[year,m,d,h,m, ConocoPhilips,Chevron,Ford,GeneralMotors,CocaCola,OccidentalPetroleum,Pepsi,ExxonMobil]
                 (year,month,day,hour,minute,conco_start,conco_end,conco_size,aa,bb,cc,ford_start,ford_end,ford_size,gg,hh,ii,jj,kk,ll,mm,nn,oo,pp,qq,rr,ss,tt,uu,zz) = row
                 if hour=="9" and minute=="30" or hour=="15" and minute=="59" or month=="12" and day=="24" and hour=="12" and minute=="59" or month=="7" and day=="3" and hour=="12" and minute=="59" or year=="2013" and month=="11" and day=="29" and hour=="12" and minute=="59" or year=="2012" and month=="11" and day=="23" and hour=="12" and minute=="59" or year=="2011" and month=="11" and day=="25" and hour=="12" and minute=="59":
                    row= (year,month,day,hour,minute,ford_start,ford_end,ford_size)
                    correlationday.write(",".join(row))
                    correlationday.write("\n")


    with open("Data\\ggg.csv", mode="rt", encoding="utf8") as ggg, \
         open(os.path.join(newpath,'correlationminute.csv'), mode="wt", encoding="utf8") as correlationminute:
        ggg = csv.reader(ggg, delimiter=",")
        for row in ggg:
                 (year,month,day,hour,minute,conco_start,conco_end,conco_size,aa,bb,cc,ford_start,ford_end,ford_size,gg,hh,ii,jj,kk,ll,mm,nn,oo,pp,qq,rr,ss,tt,uu,zz) = row
                 if year=="2011" or year=="2013":
                     row= (year,month,day,hour,minute,ford_start,ford_end,ford_size)
                     correlationminute.write(",".join(row))
                     correlationminute.write("\n")

    with open("Data\\output\\correlationday.csv", mode="rt", encoding="utf8") as correlationday:
        correlationday = csv.reader(correlationday, delimiter=",")
        op=[]  #מחיר התחלתי
        cp=[]  #מחיר סופי
        yield1=[]  #תשואה  יומית
        yield2=[]  #תשואה יומית בריבוע
        for row in correlationday:
                (year,month,day,hour,minute,ford_start,ford_end,ford_size) = row
                op.append(float(ford_start))
                cp.append(float(ford_end))
        for i in range(0,1507,2):
            yield1.append(np.log(cp[i+1]/op[i]))    #תשואה יומית כפי שהוגדרה
            yield2.append((np.log(cp[i+1]/op[i]))**2)

    with open("Data\\output\\correlationminute.csv", mode="rt", encoding="utf8") as correlationminute:
        correlationminute = csv.reader(correlationminute, delimiter=",")
        yield12=[]  #תשואה בדקה
        yield22=[]  #תשואה בדקה בריבוע
        for row in correlationminute:
                (year,month,day,hour,minute,ford_start,ford_end,ford_size) = row
                yield12.append(np.log(float(ford_end)/float(ford_start)))    #תשואה בדקה כפי שהוגדרה
                yield22.append((np.log(float(ford_end)/float(ford_start)))**2)


except FileNotFoundError:
    print("ggg.csv doesn't found")

day=[]
minute=[]
print("Ford Stock")
print("\n")
print("Q1")
print("--------------------")

for i in range(1,21):
    print("k=",i, "autcorr=","%.5f" % autocorr(yield1,i), "Validate= ", (abs(autocorr(yield1,i))>2/np.sqrt(len(yield1))))  #תשואה יומית


print("\n")
print("שאלה 1 סעיף ב")

tsplot(pd.Series(yield1), lags=20)

print("--------------------")
for i in range(1,21):
    print("k=",i, "autcorr=","%.5f" % autocorr(yield2,i), "Validate= ", (abs(autocorr(yield2,i))>2/np.sqrt(len(yield2))))  #תשואה יומית בריבוע

#correlogram(yield2)
tsplot(pd.Series(yield2), lags=20)



print("\n")
print("שאלה 1 סעיף ג")
print("--------------------")
for i in range(1,21):
    print("k=",i, "autcorr=","%.5f" % autocorr(yield12,i), "Validate= ", (abs(autocorr(yield12,i))>2/np.sqrt(len(yield12))))  #תשואה בדקה

#correlogram(yield12)
tsplot(pd.Series(yield12), lags=20)

print("\n")
print("שאלה 1 סעיף ד")
print("--------------------")
for i in range(1,21):
    print("k=",i, "autcorr=","%.5f" % autocorr(yield22,i), "Validate= ", (abs(autocorr(yield22,i))>2/np.sqrt(len(yield22))))  #תשואה בדקה בריבוע

tsplot(pd.Series(yield22), lags=20)
#correlogram(yield22)

for i in range(1,100):
    day.append(autocorr(yield1,i))
    minute.append(autocorr(yield12,i))
print(np.corrcoef(day,minute)[1,0])
