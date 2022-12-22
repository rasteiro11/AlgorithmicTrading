import pandas as pd
from pandas.core.frame import DataFrame 
import yfinance as yf
import matplotlib.pyplot as plt 

# importing stock data from yahoo finance
ticker = ["AAPL", "BA", "KO", "IBM", "DIS", "MSFT"]

stocks = yf.download("AAPL", start="2013-01-01", end="2022-12-22")

stocks.head()

stocks.tail()

stocks.info()

stocks.to_csv("stocks.csv")

stocks = pd.read_csv("stocks.csv")

stocks.head()

ticker = ["AAPL", "BA", "KO", "IBM", "DIS", "MSFT"]

stocks = yf.download(ticker, start="2013-01-01", end="2022-12-22")

stocks.head()

stocks.tail()

stocks.info()

stocks.to_csv("stocks.csv")

stocks = pd.read_csv("stocks.csv", header = [0, 1], index_col=[0], parse_dates=[0])

stocks.columns = stocks.columns.to_flat_index()

stocks.columns

stocks.columns = pd.MultiIndex.from_tuples(stocks.columns)

stocks.head()

stocks.swaplevel(axis=1).sort_index(axis=1)

# initial inspection and visualization
import pandas as pd

stocks = pd.read_csv("stocks.csv", header=[0, 1], index_col=[0],parse_dates=[0])

stocks.head()

stocks.tail()

stocks.info()

stocks.describe()

close = stocks.loc[:, "Close"].copy()

close.head()

import matplotlib.pyplot as plt 
plt.style.use("seaborn")

close.plot(figsize=(15, 8), fontsize=13)
plt.legend(fontsize=13)
plt.show()

# normalizing time series to a base value (100)
close.head()

close.iloc[0, 0]

close.loc[:, "AAPL"].div(close.iloc[0, 0]).mul(100)

close.iloc[0]

close.div(close.iloc[0]).mul(100)

close.plot(figsize=(15, 8), fontsize=13)
plt.legend(fontsize=13)
plt.show()

# the shift() mehtod

close.head()

aapl = close.loc[:, "AAPL"].copy().to_frame()

aapl.head()

aapl.shift(1)

aapl["lag1"] = aapl.shift(1)

aapl.head()

aapl["Diff"] = aapl.loc[:, "AAPL"].sub(aapl.loc[:, "lag1"])

aapl.head()

aapl["pct_change"]=  aapl.loc[:, "AAPL"].div(aapl.loc[:, "lag1"].sub(1).mul(100))

aapl.head()

# methods diff() and pct_change()
aapl.head()

aapl.AAPL.diff(1)

aapl["Diff2"] = aapl.AAPL.diff(1)

aapl.head(10)

aapl.Diff.equals(aapl.Diff2)

aapl["pct_change2"] = aapl.AAPL.pct_change(1).mul(100)

aapl.head()

aapl.AAPL.resample("BM").last().pct_change(1).mul(100)

# measuring stock performance with mean return and std of returns
import numpy as np

aapl = close.AAPL.copy().to_frame()

aapl.head()

aapl.pct_change().dropna()

ret = aapl.pct_change().dropna()
ret.head()

ret.info()

ret.plot(kind="hist", figsize=(12, 8), bins=100)
plt.show()

daily_mean_Returns = ret.mean()
print(daily_mean_Returns)

var_daily_Returns = ret.var()
print(var_daily_Returns)

std_daily_Returns = np.sqrt(var_daily_Returns)
print(std_daily_Returns)

print(ret.std())

ann_mean_Return = ret.mean() * 252
print(ann_mean_Return)

ann_var_Returns = ret.var() * 252
print(ann_var_Returns)

ann_std_Returns = np.sqrt(ann_var_Returns) * np.sqrt(252)
print(ann_var_Returns)

print(ret.std() * np.sqrt(252))


