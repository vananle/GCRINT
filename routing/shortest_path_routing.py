import itertools

import networkx as nx

from . import util


class ShortestPathRoutingSolver:

    def __init__(self, G):
        self.G = G

    def extract_utilization(self, tm):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        # initialize link load
        for u, v in G.edges:
            G[u][v]['load'] = 0
        for i, j in itertools.product(range(num_node), range(num_node)):
            path = nx.Graph()
            path.add_nodes_from(G)
            nx.add_path(path, util.shortest_path(G, i, j))
            for u, v in path.edges:
                G[u][v]['load'] += tm[i, j]
        # compute link utilization
        for u, v in G.edges:
            G[u][v]['utilization'] = G[u][v]['load'] / G[u][v]['capacity']

    def extract_utilization_v2(self, tm):
        self.extract_utilization(tm)

    def solve(self, tm):
        self.extract_utilization(tm)

    def get_path(self, i, j):
        G = self.G
        path = util.shortest_path(G, i, j)
        return
