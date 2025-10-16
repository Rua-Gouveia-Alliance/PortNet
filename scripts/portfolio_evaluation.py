import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import FormatStrFormatter

def extract_communities(graph):
    communities_dict = defaultdict(list)
    for node, data in graph.nodes(data=True):
        community_id = data["community"]
        communities_dict[community_id].append(node)

    communities = list(communities_dict.values())
    print(communities)
    return communities

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
        returns.append((portfolio, p_returns * 100, sharpe_ratio))

    return returns

def show_performance_graphs(sp500_results, ftse100_returns):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    _, sp500_returns, sp500_sharpe = zip(*sp500_results)
    sp500_top_performers = np.argsort(sp500_returns)[-3:]
    sp500_colors = np.array(['blue'] * len(sp500_results))
    sp500_colors[sp500_top_performers] = 'red'

    ax1.scatter(sp500_sharpe, sp500_returns, c=sp500_colors, s=15)
    ax1.set_title("S&P 500 portfolios")
    ax1.set_xlabel("Sharpe Ratio")
    ax1.set_ylabel("Returns (%)")

    _, ftse100_returns, ftse100_sharpe = zip(*ftse100_results)
    ftse100_top_performers = np.argsort(ftse100_returns)[-3:]
    ftse100_colors = np.array(['blue'] * len(ftse100_results))
    ftse100_colors[ftse100_top_performers] = 'red'

    ax2.scatter(ftse100_sharpe, ftse100_returns, c=ftse100_colors, s=15)
    ax2.set_title("FTSE 100 portfolios")
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_ylabel("Returns (%)")

    plt.tight_layout()
    plt.show()

def show_differences_info(sp500_results, ftse100_results):
    differences = []
    for portfolio in sp500_results:
        strategy_name = portfolio[0][6:]
        ftse100_portfolio = [p for p in ftse100_results if p[0] == f"ftse100_{strategy_name}"][0]
        differences.append([strategy_name, abs(portfolio[1] - ftse100_portfolio[1])])

    labels, diffs = zip(*differences)
    plt.figure(figsize=(20, 6))
    plt.bar(labels, diffs, color="skyblue")
    plt.xlabel("Strategy")
    plt.ylabel("Absolute return difference")
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("**************** Performance Differences ****************")
    differences.sort(key=lambda e : e[1])
    for portfolio in differences:
        print(f"{portfolio[0]}: {portfolio[1]}")

def get_index_modularity(index_id):
    modularities = {}
    graph_filenames = glob.glob(f"../data/graphs/*/{index_id}_*.gexf")
    for graph_filename in graph_filenames:
        graph = nx.read_gexf(graph_filename)
        communities = extract_communities(graph)
        modularity = nx.community.modularity(graph, communities)
        name = graph_filename[graph_filename.rfind("/")+1:graph_filename.rfind("_")]
        modularities[name] = modularity
    return modularities

def show_performance_per_modularity(sp500_results, ftse100_results):
    sp500_modularities = get_index_modularity("sp500")
    ftse100_modularities = get_index_modularity("ftse100")

    sp500_returns = []
    sp500_mods = []
    for result in sp500_results:
        name = result[0][:result[0].rfind("_", 0, result[0].rfind("_"))]
        sp500_returns.append(result[1])
        sp500_mods.append(sp500_modularities[name])

    ftse100_returns = []
    ftse100_mods = []
    for result in ftse100_results:
        name = result[0][:result[0].rfind("_", 0, result[0].rfind("_"))]
        ftse100_returns.append(result[1])
        ftse100_mods.append(ftse100_modularities[name])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    sp500_top_performers = np.argsort(sp500_returns)[-3:]
    sp500_colors = np.array(['blue'] * len(sp500_results))
    sp500_colors[sp500_top_performers] = 'red'

    ax1.scatter(sp500_mods, sp500_returns, c=sp500_colors, s=15)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.set_title("S&P 500 portfolios")
    ax1.set_xlabel("Community Modularity")
    ax1.set_ylabel("Returns (%)")

    ftse100_top_performers = np.argsort(ftse100_returns)[-3:]
    ftse100_colors = np.array(['blue'] * len(ftse100_results))
    ftse100_colors[ftse100_top_performers] = 'red'

    ax2.scatter(ftse100_mods, ftse100_returns, c=ftse100_colors, s=15)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_title("FTSE 100 portfolios")
    ax2.set_xlabel("Community Modularity")
    ax2.set_ylabel("Returns (%)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sp500_results = compute_index_portfolio_evaluation("sp500")
    ftse100_results = compute_index_portfolio_evaluation("ftse100")
    total_results = sp500_results + ftse100_results

    show_performance_per_modularity(sp500_results, ftse100_results)
    show_performance_graphs(sp500_results, ftse100_results)
    show_differences_info(sp500_results, ftse100_results)

    print("******* Portfolio Performances and Sharpe Ratios *******")
    total_results.sort(key=lambda e : e[1])
    for portfolio in total_results:
        print(f"{portfolio[0]}: {portfolio[1]}, {portfolio[2]}")
