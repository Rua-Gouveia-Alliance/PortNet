import yfinance as yf
import pandas as pd

df = pd.read_csv("sp500_survivors.csv")
tickers = df["Symbol"].to_list()

data = yf.download(tickers, period="3y", group_by="ticker")
data.to_excel("sp500_prices.xlsx")
