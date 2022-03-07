import pandas_datareader.data as web
import pandas as pd
import datetime

file1 = open('2021Q1_Top 50_of_Mutual_Funds_list.txt', 'r', encoding='utf-8')
Lines = file1.readlines()

count = 0
tickers = []
names = []

# Strips the newline character
for line in Lines:
    if line[0] == '#':
        continue

    count += 1
    ticker, name = line.split()
    tickers.append(ticker)
    names.append(name)

# training set
start = datetime.datetime(2016, 8, 19)
end = datetime.datetime(2020, 1, 15)
df = web.DataReader(tickers, 'yahoo', start, end)['Adj Close']
df.columns = names
df.to_csv('2021Q1_Top 50_of_Mutual_Funds_20160819_to_20200115.csv')


# all time dataset
start = datetime.datetime(2016, 8, 19)
end = datetime.datetime(2021, 4, 12)
df = web.DataReader(tickers, 'yahoo', start, end)['Adj Close']
df.columns = names
df.to_csv('2021Q1_Top 50_of_Mutual_Funds_20160819_to_20210412.csv')

# test set
start = datetime.datetime(2020, 1, 15)
end = datetime.datetime(2021, 1, 15)
df = web.DataReader(tickers, 'yahoo', start, end)['Adj Close']
df.columns = names
df.to_csv('2021Q1_Top 50_of_Mutual_Funds_20200115_to_20210115.csv')