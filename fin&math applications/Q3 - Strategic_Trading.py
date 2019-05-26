import numpy as np
import pandas as pd

def main():

    df = pd.read_excel('Q3b.xlsx')
    dif_avg = df['diff']
    s=100.0
    portfolio = 100.0
    bar = (np.average(dif_avg))  # Barrier - the average of difference between yields
    long = 100
    short = 100
    flag = 0
    #print(df.head())
    count = 0
    for i in range(2, len(df['year']) - 1):
        if df['diff'][i - 1] >= bar and np.maximum(df['Ch_return_d'][i - 1],
                                                         df['Ex_return_d'][i - 1]) > 0:
            if df['Ch_return_d'][i - 1] > df['Ex_return_d'][i - 1]:
                longyield = 'Ch_return_d'
                shortyield = 'Ex_return_d'
                flag = 1
            else:
                longyield = 'Ex_return_d'
                shortyield = 'Ch_return_d'
                flag = 1
        if flag == 1:
            count+=1
            portfolio = portfolio + long * df[longyield][i + 1] + short * df[shortyield][i + 1]
            long = long * df[longyield][i + 1] + long
            short = short * df[shortyield][i + 1] + short
    print((portfolio/s-1)*100,'\n Transactions:', count)

main()



