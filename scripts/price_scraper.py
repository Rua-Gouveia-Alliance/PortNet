import yfinance as yf
import pandas as pd

#df = pd.read_csv("../data/datasets/sp500_survivors.csv")
df = pd.read_csv("../data/datasets/ftse100_survivors.csv")
tickers = df["Symbol"].to_list()

data = yf.download(tickers, period="3y", group_by="ticker")

bad, good = [], []

if isinstance(data.columns, pd.MultiIndex):
    # Multi-ticker: level 0 = ticker, level 1 = field (Open/High/Low/Close/...)
    present = set(data.columns.get_level_values(0))
    for t in tickers:
        if t not in present or (t, "Close") not in data.columns:
            bad.append(t)
            continue
        close = data[(t, "Close")]
        if close.isna().any():
            bad.append(t)
        else:
            good.append(t)
    # keep only good tickers in the big frame
    if good:
        data = data.loc[:, data.columns.get_level_values(0).isin(good)]
    else:
        data = pd.DataFrame(index=pd.DatetimeIndex([]))  # nothing left
else:
    # Single-ticker fallback
    if data.empty or "Close" not in data.columns or data["Close"].isna().any():
        bad = tickers[:]  # drop it
        good = []
        data = pd.DataFrame(index=pd.DatetimeIndex([]))
    else:
        good = tickers[:]

split_index = len(data) - 252 - 1 # approximately 252 trading days in a year

data_train = data[:split_index]
data_test = data[split_index:]

#data_train.to_excel("../data/datasets/sp500_prices_train.xlsx")
#data_test.to_excel("../data/datasets/sp500_prices_test.xlsx")
data_train.to_excel("../data/datasets/ftse100_prices_train.xlsx")
data_test.to_excel("../data/datasets/ftse100_prices_test.xlsx")

if bad:
    df_filtered = df[~df["Symbol"].isin(bad)].copy()
    df_filtered.to_csv("../data/datasets/ftse100_survivors.csv", index=False)
