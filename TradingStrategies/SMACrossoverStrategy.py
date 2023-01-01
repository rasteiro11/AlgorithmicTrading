import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv("./eurusd.csv", parse_dates=["Date"], index_col="Date")

data 

sma_s = 50
sma_l = 200

data.price.rolling(50)

data["SMA_S"] = data.price.rolling(sma_s).mean()
data["SMA_L"] = data.price.rolling(sma_l).mean()

data.plot(figsize=(12, 8), title="EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize=12)
plt.legend(fontsize=12)
plt.show()

data.dropna(inplace=True)

data 

data.loc["2016"].plot(figsize=(12, 8), title="EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize=12)
plt.legend(fontsize=12)
plt.show()

data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)

data.loc["2016", ["SMA_S", "SMA_L", "position"]].plot(figsize=(12, 8), title="EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize=12, secondary_y="position")
plt.legend(fontsize=12)
plt.show()

# vectorized strategy backtesting

data

data["returns"] = np.log(data.price.div(data.price.shift(1)))
data["strategy"] = data.position.shift(1) * data["returns"]

data

data.dropna(inplace=True)

data

# absolute perfrmance
data[["returns", "strategy"]].sum()

# absolute performance
data[["returns", "strategy"]].sum().apply(np.exp)

# annualized return
data[["returns", "strategy"]].mean() * 252 

# annualized risk
data[["returns", "strategy"]].std() * np.std(252)

data["creturns"] = data["returns"].cumsum().apply(np.exp)
data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

data

data[["creturns", "cstrategy"]].plot(figsize=(12, 8), title="EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize=12, secondary_y="position")
plt.legend(fontsize=12)
plt.show()

outperf = data.cstrategy.iloc[-1] - data.creturns.iloc[-1]

outperf

# finding optimal SMA strategy
df = pd.read_csv("./eurusd.csv", parse_dates=["Date"], index_col="Date")

df

def test_strategy(SMA):
    data = df.copy()
    data["returns"] = np.log(data.price.div(data.price.shift(1)))
    data["SMA_S"] = data.price.rolling(int(SMA[0])).mean()
    data["SMA_L"] = data.price.rolling(int(SMA[1])).mean()
    data.dropna(inplace=True)

    data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
    data["strategy"] = data.position.shift(1) * data["returns"]
    data.dropna(inplace=True)

    return np.exp(data["strategy"].sum())

test_strategy((50, 200))
test_strategy((75, 150))
test_strategy((25, 252))

SMA_S_range = range(10, 50, 1)
SMA_L_range = range(100, 252, 1)

SMA_S_range

from itertools import product

list(product(SMA_S_range, SMA_L_range))

combinations = list(product(SMA_S_range, SMA_L_range))

results = []
for comb in combinations:
    results.append(test_strategy(comb))

np.max(results)

np.argmax(results)

combinations[np.argmax(results)]

many_results = pd.DataFrame(data=combinations, columns=["SMA_S", "SMA_L"])
many_results

many_results["performance"] = results
many_results

many_results.nlargest(10, "performance")

many_results.nsmallest(10, "performance")


