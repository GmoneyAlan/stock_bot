import pandas as pd 
import pandas_datareader as pdr
import datetime as dt

start_date = dt.datetime(1995,1,16)
end_date = dt.datetime(2020, 9, 1)

stock = 'AAPL'

df = pdr.DataReader(stock, 'yahoo', start_date, end_date)
#print(df.head())
df.to_csv('AAPL.csv',index=False)

