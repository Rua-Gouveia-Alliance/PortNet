import yfinance as yf
import pandas as pd

#df = pd.read_csv("../data/datasets/sp500_survivors.csv")
df = pd.read_csv("../data/datasets/ftse100_survivors.csv")
tickers = df["Symbol"].to_list()

data = yf.download(tickers, period="3y", group_by="ticker")

split_index = len(data) - 252 - 1 # approximately 252 trading days in a year

data_train = data[:split_index]
data_test = data[split_index:]

#data_train.to_excel("../data/datasets/sp500_prices_train.xlsx")
#data_test.to_excel("../data/datasets/sp500_prices_test.xlsx")
data_train.to_excel("../data/datasets/ftse100_prices_train.xlsx")
data_test.to_excel("../data/datasets/ftse100_prices_test.xlsx")
