import numpy as np
import pandas as pd
from scipy import stats

tickers = (pd.read_csv("../data/sp500_survivors.csv"))["Symbol"].to_list()
data = pd.read_excel("../data/sp500_prices.xlsx", header=[0, 1], index_col=0)

# Computing daily log returns for each asset
log_returns = {}
for ticker in tickers:
    prices = data[(ticker, "Close")].to_list()
    returns = []
    for i in range(1, len(prices)):
        returns.append(np.log(prices[i] / prices[i - 1]))
    log_returns[ticker] = returns

# Normalizing the daily log returns using min-max normalization
for ticker in tickers:
    returns = log_returns[ticker]
    max_ret = max(returns)
    min_ret = min(returns)
    diff = max_ret - min_ret
    for i in range(len(returns)):
        log_returns[ticker][i] = (returns[i] - min_ret) / diff

# Computing the distance between each asset based on the Pearson correlation
# coefficient
pcc = {}
for t1 in tickers:
    pcc[t1] = {}
    for t2 in tickers:
        corr = 1 if t1 == t2 else stats.pearsonr(log_returns[t1], log_returns[t2]).statistic
        pcc[t1][t2] = np.sqrt(2 * (1 - corr))
pd.DataFrame.from_dict(pcc).to_excel("../data/sp500_pcc_distances.xlsx")

# Computing the distance between each asset based on the Spearman correlation
# coefficient
# Since this coefficient varies between -1 and 1, we can use the same
# transformation to obtain the distance matrix
scc = {}
for t1 in tickers:
   scc[t1] = {}
   for t2 in tickers:
       corr = 1 if t1 == t2 else stats.spearmanr(log_returns[t1], log_returns[t2]).statistic
       scc[t1][t2] = np.sqrt(2 * (1 - corr))
pd.DataFrame.from_dict(scc).to_excel("../data/sp500_scc_distances.xlsx")
