import pandas as pd
import distances
import communities

def split_data_into_intervals(data, intervals):
    """
    Given a dataset of historically price data and a list of tuples with time
    intervals, split the dataset into multiple datasets containing only the
    prices for a given interval
    """
    splits = []
    for interval in intervals:
        splits.append(data.loc[interval[0]:interval[1]])
    return splits

def get_monthly_intervals(data):
    min_date = data.index.to_list()[0]
    max_date = data.index.to_list()[-1]
    current_date = min_date
    intervals = []

    while (current_date + pd.offsets.MonthEnd(1)) < max_date:
        intervals.append((current_date, current_date + pd.offsets.MonthEnd(1)))
        current_date += pd.offsets.MonthEnd(1)

    intervals.append((current_date, max_date))
    return intervals

def get_monthly_log_returns(tickers, splits):
    log_returns = []
    for split in splits:
        log_returns.append(distances.get_normalized_daily_log_returns(tickers, split))
    return log_returns

def get_monthly_pcc_distances(tickers, returns):
    pcc_distances = []
    for split in returns:
        pcc_distances.append(distances.get_pcc_distances(tickers, split))
    return pcc_distances

def get_monthly_scc_distances(tickers, returns):
    scc_distances = []
    for split in returns:
        scc_distances.append(distances.get_scc_distances(tickers, split))
    return scc_distances

def get_monthly_msts(tickers, dists):
    msts = []
    for distance in dists:
        msts.append(communities.get_mst(tickers, distance))
    return msts

def get_monthly_louvain(msts):
    comms = []
    for mst in msts:
        comms.append(communities.get_louvain(mst))
    return comms

def get_monthly_girvan_newman(msts, sizes):
    comms = []
    for mst, size in zip(msts, sizes):
        comms.append(communities.get_girvan_newman(mst, size))
    return comms

def get_cooccurrence_matrix(tickers, monthly_communities):
    cooccurrence = {}
    for t1 in tickers:
        cooccurrence[t1] = {}
        for t2 in tickers:
            cooccurrence[t1][t2] = 0

    # Calculating frequencies
    count = len(monthly_communities)
    for comms in monthly_communities:
        for community in comms:
            for t1 in community:
                for t2 in community:
                    cooccurrence[t1][t2] += 1 / count

    return pd.DataFrame.from_dict(cooccurrence)

if __name__ == "__main__":
    tickers = (pd.read_csv("../data/datasets/sp500_survivors.csv"))["Symbol"].to_list()
    data = pd.read_excel("../data/datasets/sp500_prices.xlsx", header=[0, 1], index_col=0)
    intervals = get_monthly_intervals(data)
    splits = split_data_into_intervals(data, intervals)
    monthly_log_returns = get_monthly_log_returns(tickers, splits)

    # PCC
    monthly_pcc_distances = get_monthly_pcc_distances(tickers, monthly_log_returns)
    monthly_pcc_msts = get_monthly_msts(tickers, monthly_pcc_distances)

    # Louvain Co-Occurrence
    monthly_pcc_louvain = get_monthly_louvain(monthly_pcc_msts)
    pcc_louvain_cooccurrence = get_cooccurrence_matrix(tickers, monthly_pcc_louvain)
    pcc_louvain_cooccurrence.to_excel("../data/auxiliary/sp500_pcc_louvain_cooccurrence.xlsx")

    # Girvan-Newman Co-Occurrence
    monthly_pcc_girvan_newman = get_monthly_girvan_newman(monthly_pcc_msts, [len(c) for c in monthly_pcc_louvain])
    pcc_girvan_newman_cooccurrence = get_cooccurrence_matrix(tickers, monthly_pcc_girvan_newman)
    pcc_girvan_newman_cooccurrence.to_excel("../data/auxiliary/sp500_pcc_girvan_newman_cooccurrence.xlsx")

    # SCC
    monthly_scc_distances = get_monthly_scc_distances(tickers, monthly_log_returns)
    monthly_scc_msts = get_monthly_msts(tickers, monthly_scc_distances)

    # Louvain Co-Occurrence
    monthly_scc_louvain = get_monthly_louvain(monthly_scc_msts)
    scc_louvain_cooccurrence = get_cooccurrence_matrix(tickers, monthly_scc_louvain)
    scc_louvain_cooccurrence.to_excel("../data/auxiliary/sp500_scc_louvain_cooccurrence.xlsx")

    # Girvan-Newman Co-Occurrence
    monthly_scc_girvan_newman = get_monthly_girvan_newman(monthly_scc_msts, [len(c) for c in monthly_scc_louvain])
    scc_girvan_newman_cooccurrence = get_cooccurrence_matrix(tickers, monthly_scc_girvan_newman)
    scc_girvan_newman_cooccurrence.to_excel("../data/auxiliary/sp500_scc_girvan_newman_cooccurrence.xlsx")
