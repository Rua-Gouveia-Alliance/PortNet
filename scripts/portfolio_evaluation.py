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

    return returns

if __name__ == "__main__":
    sp500_results = compute_index_portfolio_evaluation("sp500")
    ftse100_results = compute_index_portfolio_evaluation("ftse100")
    total_results = sp500_results + ftse100_results

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    _, sp500_returns, sp500_sharpe = zip(*sp500_results)
    sp500_colors = np.random.rand(len(sp500_returns), 3)

    ax1.scatter(sp500_sharpe, sp500_returns, c=sp500_colors, s=15)
    ax1.set_title("S&P 500 portfolios")
    ax1.set_xlabel("Sharpe Ratio")
    ax1.set_ylabel("Returns (%)")

    _, ftse100_returns, ftse100_sharpe = zip(*ftse100_results)
    ftse100_colors = np.random.rand(len(ftse100_returns), 3)

    ax2.scatter(ftse100_sharpe, ftse100_returns, c=ftse100_colors, s=15)
    ax2.set_title("FTSE 100 portfolios")
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_ylabel("Returns (%)")

    _, total_returns, total_sharpe = zip(*total_results)
    colors = list(sp500_colors) + list(ftse100_colors)

    ax3.scatter(total_sharpe, total_returns, c=colors, s=15)
    ax3.set_title("All examined portfolios")
    ax3.set_xlabel("Sharpe Ratio")
    ax3.set_ylabel("Returns (%)")

    plt.tight_layout()
    plt.show()

    total_results.sort(key=lambda e : e[1])
    for portfolio in total_results:
        print(f"{portfolio[0]}: {portfolio[1]}, {portfolio[2]}")
