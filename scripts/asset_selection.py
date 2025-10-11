import glob
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA as SklearnPCA
from distances import get_normalized_daily_log_returns

TOTAL_ASSETS = 25
PCA_N_COMPONENTS = 1 # focus on the first principal component for ranking, as it captures market-wide movement

#TRAINING_PRICES_FILE = "../data/datasets/sp500_prices_train.xlsx"
#TICKERS_FILE = "../data/datasets/sp500_survivors.csv"
TRAINING_PRICES_FILE = "../data/datasets/ftse100_prices_train.xlsx"
TICKERS_FILE = "../data/datasets/ftse100_survivors.csv"

def calculate_pca_scores(tickers):
    prices_full = pd.read_excel(TRAINING_PRICES_FILE, header=[0, 1], index_col=0)

    log_returns_dict = get_normalized_daily_log_returns(tickers, prices_full)
    normalized_returns_df = pd.DataFrame(log_returns_dict).dropna(axis=0, how='any')

    # compute correlation matrix
    correlation_matrix = normalized_returns_df.corr()

    # fit PCA
    pca_model = SklearnPCA(n_components=PCA_N_COMPONENTS)
    pca_model.fit(correlation_matrix)

    # extract scores
    pca_scores = pd.Series(
        np.abs(pca_model.components_[0]),
        index=correlation_matrix.index
    )

    return pca_scores.to_dict()


def calculate_centrality_metrics(mst_graph):
    # degree centrality
    degree_centrality = nx.degree_centrality(mst_graph)

    # closeness centrality
    temp_graph = mst_graph.copy()
    for u, v, data in temp_graph.edges(data=True):
        # revert similarity back to distance for closeness path finding
        data['distance'] = 2 - data['weight']
    closeness_centrality = nx.closeness_centrality(temp_graph, distance='distance')

    return degree_centrality, closeness_centrality


def select_assets(stabilized_graph, metric_scores, total_assets=TOTAL_ASSETS):
    communities = {}
    for node, data in stabilized_graph.nodes(data=True):
        # Ensure the 'community' attribute is present and handle string vs int IDs
        comm_id = str(data.get("community")) if data.get("community") is not None else None
        if comm_id is not None:
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)

    Q = len(communities)
    if Q == 0:
        return # no communities

    # determine asset allocation (as per paper)
    base_per_comm = total_assets // Q
    remainder = total_assets % Q
    sorted_comm_ids = sorted(communities, key=lambda id: len(communities[id]), reverse=True)

    allocation = {}
    for i, comm_id in enumerate(sorted_comm_ids):
        allocation[comm_id] = base_per_comm + (1 if i < remainder else 0)

    # sort assets for each community
    scored_communities = {}
    for comm_id, assets in communities.items():
        scored_communities[comm_id] = [
            (asset, metric_scores.get(asset, -1)) for asset in assets
        ]
        # Sort ascending by score
        scored_communities[comm_id].sort(key=lambda x: x[1])

    # select assets for each range
    all_strategies = {
        "min": [], # lowest scores
        "med": [], # median score
        "max": []  # highest scores
    }

    for comm_id, asset_list in scored_communities.items():
        k = allocation[comm_id] # Number of assets to select from this community
        N_comm = len(asset_list)

        # min selection (lowest k scores)
        all_strategies["min"].extend([asset for asset, score in asset_list[:k]])

        # max selection (highest k scores)
        all_strategies["max"].extend([asset for asset, score in asset_list[-k:]])

        # med selection (k scores around the median)
        mid_start = max(0, N_comm // 2 - k // 2)
        mid_end = mid_start + k

        all_strategies["med"].extend([asset for asset, _ in asset_list[mid_start:mid_end]])

    return all_strategies

def generate_portfolios(graph_filename, tickers):
    graph = nx.read_gexf(graph_filename)
    pca_scores = calculate_pca_scores(tickers)
    degree_scores, closeness_scores = calculate_centrality_metrics(graph)

    # PCA-Based portfolios
    pca_portfolios = select_assets(graph, pca_scores)
    # Degree Centrality (CD) portfolios
    cd_portfolios = select_assets(graph, degree_scores)
    # Closeness Centrality (CC) portfolios
    cc_portfolios = select_assets(graph, closeness_scores)

    return pca_portfolios, cd_portfolios, cc_portfolios

if __name__ == "__main__":
    df_tickers = pd.read_csv(TICKERS_FILE)
    tickers = df_tickers["Symbol"].to_list()
    #graph_filenames = glob.glob("../data/graphs/*/sp500_*.gexf")
    graph_filenames = glob.glob("../data/graphs/*/ftse100_*.gexf")

    portfolios = {}
    for filename in graph_filenames:
        pca_portfolios, cd_portfolios, cc_portfolios = generate_portfolios(filename, tickers)

        name = filename[filename.rfind("/")+1:filename.rfind("_")]
        metric_names = ["pca", "cd", "cc"]
        portfolio_dictionaries = [pca_portfolios, cd_portfolios, cc_portfolios]
        for i in range(len(metric_names)):
            for range_name, assets in portfolio_dictionaries[i].items():
                strategy_name = f"{name}_{metric_names[i]}_{range_name}"
                portfolios[strategy_name] = assets

    portfolios = pd.DataFrame.from_dict(portfolios)
    #portfolios.to_excel("../data/auxiliary/sp500_portfolios.xlsx")
    portfolios.to_excel("../data/auxiliary/ftse100_portfolios.xlsx")
