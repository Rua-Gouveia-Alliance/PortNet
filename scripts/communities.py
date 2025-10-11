import pandas as pd
import networkx as nx
import itertools

def get_mst(tickers, distances):
    """
    Given a distance matrix, computes the corresponding MST and inverts its
    weights. The weight inversion is required because of how the community
    finding algorithms work.
    Our weights represent distances, so lower weights are better, which is what
    we want for the MST computation. However, the community finding algorithms
    interpret weights as a measure of similarity, i.e., bigger weights are
    better, so we must invert them before perfoming community detection.
    Weights are between 0 and 2, so 'weight = 2 - weight' does this
    inversion.
    """
    G = nx.Graph()
    G.add_nodes_from(tickers)
    for t1 in tickers:
        for t2 in tickers:
            if t1 == t2:
                break
            G.add_edge(t1, t2, weight=distances.at[t1, t2])
    mst = nx.minimum_spanning_tree(G)

    # Weight inversion
    for u, v in mst.edges():
        mst[u][v]["weight"] = 2 - mst[u][v]["weight"]

    return mst

def get_louvain(mst):
    """ Given an MST, computes its partitioning according to the Louvain method """
    return nx.community.louvain_communities(mst)

def get_girvan_newman(mst):
    """ Given an MST, computes its partitioning according to the Girvan-Newman algorithm """
    communities = list(nx.community.girvan_newman(mst))
    best = [communities[0], nx.community.modularity(mst, communities[0])]

    for community in communities:
        modularity = nx.community.modularity(mst, community)
        if modularity > best[1]:
            best = [community, modularity]
    return best[0]

def get_communities_graph(mst, communities):
    """
    Given a graph and a list of communities, sets each node 'community'
    attribute to the appropriate value
    """
    G = mst.copy()
    community_id = 0

    for community in communities:
        for asset in community:
            G.nodes[asset]["community"] = community_id
        community_id += 1

    return G

def print_communities(communities):
    """ Prints information about a set of communities """
    print("Number of communities: " + str(len(communities)))
    print("Sizes:")
    for community in communities:
        print(len(community))

def compute_index_communities(index_id):
    tickers = (pd.read_csv(f"../data/datasets/{index_id}_survivors.csv"))["Symbol"].to_list()

    pcc_distances = pd.read_excel(f"../data/auxiliary/{index_id}_pcc_distances.xlsx", header=0, index_col=0)
    pcc_lv_co = pd.read_excel(f"../data/auxiliary/{index_id}_pcc_louvain_cooccurrence.xlsx", header=0, index_col=0)
    pcc_gn_co = pd.read_excel(f"../data/auxiliary/{index_id}_pcc_girvan_newman_cooccurrence.xlsx", header=0, index_col=0)

    scc_distances = pd.read_excel(f"../data/auxiliary/{index_id}_scc_distances.xlsx", header=0, index_col=0)
    scc_lv_co = pd.read_excel(f"../data/auxiliary/{index_id}_scc_louvain_cooccurrence.xlsx", header=0, index_col=0)
    scc_gn_co = pd.read_excel(f"../data/auxiliary/{index_id}_scc_girvan_newman_cooccurrence.xlsx", header=0, index_col=0)

    pcc_mst = get_mst(tickers, pcc_distances)
    pcc_lv_co_mst = get_mst(tickers, pcc_lv_co)
    pcc_gn_co_mst = get_mst(tickers, pcc_gn_co)

    scc_mst = get_mst(tickers, scc_distances)
    scc_lv_co_mst = get_mst(tickers, scc_lv_co)
    scc_gn_co_mst = get_mst(tickers, scc_gn_co)

    # Computing the standard PCC graph partitioned using Louvain
    pcc_louvain = get_louvain(pcc_mst)
    pcc_louvain_G = get_communities_graph(pcc_mst, pcc_louvain)
    nx.write_gexf(pcc_louvain_G, f"../data/graphs/pcc_louvain/{index_id}_pcc_louvain_graph.gexf")

    # Computing the standard PCC graph partitioned using Girvan-Newman
    pcc_girvan_newman = get_girvan_newman(pcc_mst)
    pcc_girvan_newman_G = get_communities_graph(pcc_mst, pcc_girvan_newman)
    nx.write_gexf(pcc_girvan_newman_G, f"../data/graphs/pcc_girvan_newman/{index_id}_pcc_girvan_newman_graph.gexf")

    # Computing the Louvain co-occurrence PCC graph partitioned using Louvain
    pcc_lv_co_louvain = get_louvain(pcc_lv_co_mst)
    pcc_lv_co_louvain_G = get_communities_graph(pcc_lv_co_mst, pcc_lv_co_louvain)
    nx.write_gexf(pcc_lv_co_louvain_G, f"../data/graphs/pcc_lv_co_louvain/{index_id}_pcc_lv_co_louvain_graph.gexf")

    # Computing the Louvain co-occurrence PCC graph partitioned using Girvan-Newman
    pcc_lv_co_girvan_newman = get_girvan_newman(pcc_lv_co_mst)
    pcc_lv_co_girvan_newman_G = get_communities_graph(pcc_lv_co_mst, pcc_lv_co_girvan_newman)
    nx.write_gexf(pcc_lv_co_girvan_newman_G, f"../data/graphs/pcc_lv_co_girvan_newman/{index_id}_pcc_lv_co_girvan_newman_graph.gexf")

    # Computing the Girvan-Newman co-occurrence PCC graph partitioned using Louvain
    pcc_gn_co_louvain = get_louvain(pcc_gn_co_mst)
    pcc_gn_co_louvain_G = get_communities_graph(pcc_gn_co_mst, pcc_gn_co_louvain)
    nx.write_gexf(pcc_gn_co_louvain_G, f"../data/graphs/pcc_gn_co_louvain/{index_id}_pcc_gn_co_louvain_graph.gexf")

    # Computing the Girvan-Newman co-occurrence PCC graph partitioned using Girvan-Newman
    pcc_gn_co_girvan_newman = get_girvan_newman(pcc_gn_co_mst)
    pcc_gn_co_girvan_newman_G = get_communities_graph(pcc_gn_co_mst, pcc_gn_co_girvan_newman)
    nx.write_gexf(pcc_gn_co_girvan_newman_G, f"../data/graphs/pcc_gn_co_girvan_newman/{index_id}_pcc_gn_co_girvan_newman_graph.gexf")

    # Computing the standard SCC graph partitioned using Louvain
    scc_louvain = get_louvain(scc_mst)
    scc_louvain_G = get_communities_graph(scc_mst, scc_louvain)
    nx.write_gexf(scc_louvain_G, f"../data/graphs/scc_louvain/{index_id}_scc_louvain_graph.gexf")

    # Computing the standard SCC graph partitioned using Girvan-Newman
    scc_girvan_newman = get_girvan_newman(scc_mst)
    scc_girvan_newman_G = get_communities_graph(scc_mst, scc_girvan_newman)
    nx.write_gexf(scc_girvan_newman_G, f"../data/graphs/scc_girvan_newman/{index_id}_scc_girvan_newman_graph.gexf")

    # Computing the Louvain co-occurrence SCC graph partitioned using Louvain
    scc_lv_co_louvain = get_louvain(scc_lv_co_mst)
    scc_lv_co_louvain_G = get_communities_graph(scc_lv_co_mst, scc_lv_co_louvain)
    nx.write_gexf(scc_lv_co_louvain_G, f"../data/graphs/scc_lv_co_louvain/{index_id}_scc_lv_co_louvain_graph.gexf")

    # Computing the Louvain co-occurrence SCC graph partitioned using Girvan-Newman
    scc_lv_co_girvan_newman = get_girvan_newman(scc_lv_co_mst)
    scc_lv_co_girvan_newman_G = get_communities_graph(scc_lv_co_mst, scc_lv_co_girvan_newman)
    nx.write_gexf(scc_lv_co_girvan_newman_G, f"../data/graphs/scc_lv_co_girvan_newman/{index_id}_scc_lv_co_girvan_newman_graph.gexf")

    # Computing the Girvan-Newman co-occurrence SCC graph partitioned using Louvain
    scc_gn_co_louvain = get_louvain(scc_gn_co_mst)
    scc_gn_co_louvain_G = get_communities_graph(scc_gn_co_mst, scc_gn_co_louvain)
    nx.write_gexf(scc_gn_co_louvain_G, f"../data/graphs/scc_gn_co_louvain/{index_id}_scc_gn_co_louvain_graph.gexf")

    # Computing the Girvan-Newman co-occurrence SCC graph partitioned using Girvan-Newman
    scc_gn_co_girvan_newman = get_girvan_newman(scc_gn_co_mst)
    scc_gn_co_girvan_newman_G = get_communities_graph(scc_gn_co_mst, scc_gn_co_girvan_newman)
    nx.write_gexf(scc_gn_co_girvan_newman_G, f"../data/graphs/scc_gn_co_girvan_newman/{index_id}_scc_gn_co_girvan_newman_graph.gexf")

if __name__ == "__main__":
    compute_index_communities("sp500")
    compute_index_communities("ftse100")
