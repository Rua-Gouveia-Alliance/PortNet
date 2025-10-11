import numpy as np
import pandas as pd
from scipy import stats

def get_normalized_daily_log_returns(tickers, data):
    """
    Given a list of tickers and their prices over a time period, compute the
    normalized daily log returns for each asset
    """
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

    return log_returns

def get_pcc_distances(tickers, returns):
    """
    Computes a distance matrix for assets using the Pearson correlation
    coefficient
    """
    pcc = {}
    for t1 in tickers:
        pcc[t1] = {}
        for t2 in tickers:
            corr = 1 if t1 == t2 else stats.pearsonr(returns[t1], returns[t2]).statistic
            pcc[t1][t2] = np.sqrt(2 * (1 - corr))
    return pd.DataFrame.from_dict(pcc)

def get_scc_distances(tickers, returns):
    """
    Computes a distance matrix for assets using the Spearman correlation
    coefficient
    """
    # Since this coefficient varies between -1 and 1, we can use the same
    # transformation to obtain the distance matrix
    scc = {}
    for t1 in tickers:
        scc[t1] = {}
        for t2 in tickers:
            corr = 1 if t1 == t2 else stats.spearmanr(returns[t1], returns[t2]).statistic
            scc[t1][t2] = np.sqrt(2 * (1 - corr))
    return pd.DataFrame.from_dict(scc)

def compute_index_distances(index_id):
    tickers = (pd.read_csv(f"../data/datasets/{index_id}_survivors.csv"))["Symbol"].to_list()
    data = pd.read_excel(f"../data/datasets/{index_id}_prices_train.xlsx", header=[0, 1], index_col=0)
    log_returns = get_normalized_daily_log_returns(tickers, data)

    pcc_distances = get_pcc_distances(tickers, log_returns)
    scc_distances = get_scc_distances(tickers, log_returns)

    pcc_distances.to_excel(f"../data/auxiliary/{index_id}_pcc_distances.xlsx")
    scc_distances.to_excel(f"../data/auxiliary/{index_id}_scc_distances.xlsx")

if __name__ == "__main__":
    compute_index_distances("sp500")
    compute_index_distances("ftse100")
