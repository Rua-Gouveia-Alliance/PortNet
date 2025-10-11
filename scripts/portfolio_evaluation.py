import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_annualized_log_return_vol(portfolio, prices):
    daily_returns = []
    for asset in list(portfolio):
        daily_prices = list(prices[asset]["Close"])
        for i in range(1, len(daily_prices)):
            daily_returns.append(np.log(daily_prices[i] / daily_prices[i - 1]))
    return np.std(daily_returns) * np.sqrt(365)

def get_returns(portfolio, prices):
    pD = 0
    p0 = 0
    for asset in list(portfolio):
        pD += prices[asset]["Close"].iloc[-1]
        p0 += prices[asset]["Close"].iloc[0]
    return (pD - p0) / p0

def get_sharpe_ratio(portfolio, prices, returns):
    risk_free_rate = 3.86 / 100 # Annualized 13-week U.S. Treasury bill rate
    vol = get_annualized_log_return_vol(portfolio, prices)
    return (returns - risk_free_rate) / vol

def compute_index_portfolio_evaluation(index_id):
    prices = pd.read_excel(f"../data/datasets/{index_id}_prices_test.xlsx", header=[0, 1], index_col=0)
    portfolios = pd.read_excel(f"../data/auxiliary/{index_id}_portfolios.xlsx", header=0, index_col=0)
    returns = []

    for portfolio in list(portfolios):
        p_returns = get_returns(portfolios[portfolio], prices)
        sharpe_ratio = get_sharpe_ratio(portfolios[portfolio], prices, p_returns)
        returns.append((portfolio, p_returns, sharpe_ratio))

    returns.sort(key=lambda e : e[1])
    for portfolio in returns:
        print(f"{portfolio[0]}: {portfolio[1]}, {portfolio[2]}")

    _, rets, sharpe = zip(*returns)
    colors = np.random.rand(len(rets), 3)
    plt.scatter(sharpe, rets, c=colors, s=15)
    plt.title("All examined portfolios")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Returns (%)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    compute_index_portfolio_evaluation("sp500")
    compute_index_portfolio_evaluation("ftse")
