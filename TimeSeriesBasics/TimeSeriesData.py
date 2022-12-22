import pandas as pd

# importing time series data from csv-files 
temp = pd.read_csv("temp.csv", parse_dates=["datetime"], index_col=["datetime"])

temp.head()

temp.info()

temp.loc["2013-01-01 00:00:00":"2013-01-01 03:00:00"]

temp.index

# converting strings to datetime objects with pd.to_datetime()
temp = pd.read_csv("temp.csv")

temp.head()

temp.datetime

temp.info()

pd.to_datetime(temp.datetime)

temp = temp.set_index(pd.to_datetime(temp.datetime)).drop(["datetime"], axis=1)

temp.head()

temp.index

pd.to_datetime("2015-05-20")

pd.to_datetime("2015-05-20 10:30:20")

pd.to_datetime("2015/05/20")

pd.to_datetime("2015 05 20")

pd.to_datetime("2015 May 20")

pd.to_datetime("2015 20th May")

pd.to_datetime(["2015 05 20", "2016 12 20"])

pd.to_datetime(["2015 05 20", "2016 12 20", "THIS DATE DOES NOT EXISTS"], errors="coerce")

# indexing and slicing time series 

temp = pd.read_csv("temp.csv")

temp = temp.set_index(pd.to_datetime(temp.datetime)).drop(["datetime"], axis=1)

temp.loc["2015"]

temp.loc["2015":"2017"]

temp.loc["2013-01-01 01:00:00"]

temp.loc[["2015-05-20 10:00:00", "2015-05-20 12:00:00"]]

two_timestamps = pd.to_datetime(["2015-05-20 10:00:00", "2015-05-20 12:00:00"])

temp.loc[two_timestamps]

# downsampling timeseries using resample()
import matplotlib.pyplot as plt 

temp = temp.set_index(pd.to_datetime(temp.datetime)).drop(["datetime"], axis=1)

temp.info()

list(temp.resample("D"))[0][1]

temp.resample("D").first()

temp.resample("2H").first()

temp.resample("W").mean()

temp.resample("W-Wed").mean()

temp.resample("MS").mean()

temp.resample("Q").mean()

temp.resample("Q-Feb").mean()

temp.resample("Y").mean()

temp.resample("YS").mean()

# Exercises 
import pandas as pd

# import csv "aapl_fb.csv" and parse "Date" column to datetime and set "Date" column as index
stocks = pd.read_csv("aapl_fb.csv", parse_dates=["Date"], index_col=["Date"])

stocks.head()

stocks.info()

stocks.describe()

# get AAPL stock price at May 13 2016
stocks.loc["May-13-2016"]["AAPL"]

# select stock prices in Sep 2017 and get AAPL mean
stocks.loc["Sep-2017"].mean()["AAPL"]

# create a timestamp for August 10 2019
pd.to_datetime("Aug 10 2019")
pd.to_datetime("Aug-10-2019")
pd.to_datetime("Aug/10/2019")

# change frequency of stocks to monthly and display last traded stock price from FB in March 2015
stocks.resample("M").last().loc["2015 Mar"]["FB"]

# calculate monthly mean stock prices and get average stock price for FB in April 2015
stocks.resample("M").mean().loc["2015 Apr"]["FB"]

stocks_m = stocks.resample("M").mean()
stocks_m.head()

# FB price in the fourth friday in January 2015   
stocks.resample("W-Fri").last().iloc[3].loc["FB"]


