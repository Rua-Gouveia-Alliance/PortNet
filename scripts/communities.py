import pandas as pd
import networkx as nx
import itertools

# TODO: Discutir com o stor usar só MSTs em vez da rede completa
# TODO: Inverter weights da tree para o Louvain????

def get_mst(tickers, distances):
    """ Given a distance matrix, computes the corresponding MST """
    G = nx.Graph()
    G.add_nodes_from(tickers)
    for t1 in tickers:
        for t2 in tickers:
            if t1 == t2:
                break
            G.add_edge(t1, t2, weight=distances.at[t1, t2])
    return nx.minimum_spanning_tree(G)

def get_louvain(mst):
    """ Given an MST, computes its partitioning according to the Louvain algorithm """
    return nx.community.louvain_communities(mst)

def get_girvan_newman(mst, size):
    """ Given an MST, computes a partition of size `size` using the Girvan-Newman algorithm """
    *_, girvan_newman = itertools.takewhile(lambda c: len(c) <= size, nx.community.girvan_newman(mst))
    return girvan_newman

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

if __name__ == "__main__":
    tickers = (pd.read_csv("../data/datasets/sp500_survivors.csv"))["Symbol"].to_list()
    pcc_distances = pd.read_excel("../data/auxiliary/sp500_pcc_distances.xlsx", header=0, index_col=0)
    scc_distances = pd.read_excel("../data/auxiliary/sp500_scc_distances.xlsx", header=0, index_col=0)

    pcc_mst = get_mst(tickers, pcc_distances)
    scc_mst = get_mst(tickers, scc_distances)

    # Computing the PCC graph partitioned using Louvain
    pcc_louvain = get_louvain(pcc_mst)
    pcc_louvain_G = get_communities_graph(pcc_mst, pcc_louvain)
    nx.write_gexf(pcc_louvain_G, "../data/graphs/pcc_louvain/pcc_louvain_graph.gexf")

    # Computing the PCC graph partitioned using Girvan-Newman
    pcc_girvan_newman = get_girvan_newman(pcc_mst, len(pcc_louvain))
    pcc_girvan_newman_G = get_communities_graph(pcc_mst, pcc_girvan_newman)
    nx.write_gexf(pcc_girvan_newman_G, "../data/graphs/pcc_girvan_newman/pcc_girvan_newman_graph.gexf")

    # Computing the SCC graph partitioned using Louvain
    scc_louvain = get_louvain(scc_mst)
    scc_louvain_G = get_communities_graph(scc_mst, scc_louvain)
    nx.write_gexf(scc_louvain_G, "../data/graphs/scc_louvain/scc_louvain_graph.gexf")

    # Computing the SCC graph partitioned using Girvan-Newman
    scc_girvan_newman = get_girvan_newman(scc_mst, len(scc_louvain))
    scc_girvan_newman_G = get_communities_graph(scc_mst, scc_girvan_newman)
    nx.write_gexf(scc_girvan_newman_G, "../data/graphs/scc_girvan_newman/scc_girvan_newman_graph.gexf")
