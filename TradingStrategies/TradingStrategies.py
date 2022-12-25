import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.style.use("seaborn")

df = pd.read_csv("./eurusd.csv", parse_dates=["Date"], index_col=["Date"])

df.info()

df.plot(figsize=(12, 8), title="EUR/USD", fontsize=12)
plt.show()

df["returns"] = np.log(df.div(df.shift(1)))

df.head()

# simple buy and hold "Strategy" (not really a strategy)
df.dropna(inplace=True)

df.returns.hist(bins=100, figsize=(12, 8))
plt.title("EUR/USD returns")
plt.show()

df.returns.sum()

np.exp(df.returns.sum())

df.price[-1] / df.price[0]

df.returns.cumsum().apply(np.exp)

df["creturns"] = df.returns.cumsum().apply(np.exp)

df.dropna(inplace=True)

print(df)

df.creturns.iloc[-1]

df.returns.sum()

df.describe()

df.returns.mean() * 252

df.returns.std() * np.sqrt(252)

df["cummax"] = df.creturns.cummax()

df[["creturns", "cummax"]].dropna().plot(figsize=(12, 8), title="EUR/USD - max drawdown", fontsize=15)
plt.show()

drawdown = df["cummax"] - df["creturns"]
print(drawdown)

drawdown.max()

drawdown.idxmax()

