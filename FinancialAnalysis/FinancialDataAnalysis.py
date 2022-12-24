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

# financial tiem series - return and risk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

stocks = pd.read_csv("./stocks.csv", header=[0, 1], index_col=[0],parse_dates=[0])

stocks.head()

close = stocks.loc[:, "Close"].copy()

close.pct_change().dropna()

ret = close.pct_change().dropna()

ret.head()

summary = ret.describe().T.loc[:, ["mean", "std"]]

print(summary)

summary["mean"] = summary["mean"]*252
summary["std"] = summary["std"]*np.sqrt(252)

summary.plot.scatter(x="std", y="mean", figsize=(12, 8), s = 50,fontsize=15)
for i in summary.index:
    plt.annotate(i ,xy=(summary.loc[i, "std"]+0.002, summary.loc[i, "mean"]+ 0.002), size=15)

plt.xlabel("ann. Risk(std)", fontsize=15)
plt.ylabel("ann. Return", fontsize=15)
plt.title("Risk/Return", fontsize=20)
plt.show()

# financial time series - covariance and correlation
ret.head()

ret.cov()

ret.corr()

import seaborn as sns

plt.figure(figsize=(12, 8))
sns.set(font_scale=1.4)
sns.heatmap(ret.corr(), cmap="Reds", annot=True, annot_kws={"size": 15}, vmax=0.6)
plt.show()

# Exercise 2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")

# load us stocks parsing dates, setting headers and tiemstamp indices
stocks = pd.read_csv("us_stocks.csv", parse_dates=[0], index_col=0, header=[0, 1])

stocks.head()

# get Adj Close stock prices from "2015-12-31" to "2018-12-31"
stocks = stocks.loc["2015-12-31":"2018-12-31", "Adj Close"]

stocks.head()

stocks.info()

# show adjusted close prices
stocks.plot(figsize=(12, 8))
plt.show()

# normalize stock prices with base 100
first_row = stocks.iloc[0, :]
stocks.div(first_row).mul(100).plot(figsize=(12, 8))
plt.show()

# resampling stock prices to monthly rate
stocks_m = stocks.resample("M").last()

stocks_m.head()

# calculate monthly returns
ret = stocks_m.pct_change().dropna()

ret.head()

# calculate mean and std of monthly returns
summary = ret.describe().T.loc[:, ["mean", "std"]]

# annualize mean and std 
summary["mean"] = summary["mean"]*12
summary["std"] = summary["std"]*np.sqrt(12)

summary.head()
summary.plot(kind = "scatter", x = "std", y = "mean", figsize = (12, 8), s = 50, fontsize = 15, xlim = (0.1, 0.3), ylim = (0, 0.25))

for i in summary.index:
    plt.annotate(i, xy=(summary.loc[i, "std"]+0.002, summary.loc[i, "mean"]+0.002), size = 15)

plt.xlabel("ann. Risk(std)", fontsize = 15)
plt.ylabel("ann. Return", fontsize = 15)
plt.title("Risk/Return", fontsize = 20)
plt.show()

# get stock correlation between stock returns
ret.corr()

plt.figure(figsize=(12,8))
sns.set(font_scale=1.0)
sns.heatmap(ret.corr(), cmap = "RdYlGn", annot = True, vmin = -0.5, vmax = 0.6, center = 0)
plt.show()
