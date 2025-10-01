import yfinance as yf
import pandas as pd

df = pd.read_csv("../data/sp500_survivors.csv")
tickers = df["Symbol"].to_list()

data = yf.download(tickers, period="3y", group_by="ticker")
data.to_csv("../data/ticker_price.csv")
