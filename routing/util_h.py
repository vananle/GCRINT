import itertools

import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_traffic_matrix(dataset='abilene_tm', timestep=0):
    tm = loadmat('../../data/data/{}.mat'.format(dataset))['X'][timestep, :]
    num_node = int(np.sqrt(tm.shape[0]))
    tm = tm.reshape(num_node, num_node)
    return tm


def load_traffic_matrix_max(dataset='abilene_tm', start=0, end=10):
    tm = loadmat('../../data/data/{}.mat'.format(dataset))['X'][start:end, :]
    num_node = int(np.sqrt(tm.shape[1]))
    tm = tm.reshape(-1, num_node, num_node)
    return tm


def load_network_topology(dataset='abilene'):
    # initialize graph
    G = nx.Graph()
    # load node data from csv
    df = pd.read_csv('../../data/topo/{}_node.csv'.format(dataset), delimiter=' ')
    for i, row in df.iterrows():
        G.add_node(i, label=row.label, pos=(row.x, row.y))
    # load edge data from csv
    df = pd.read_csv('../../data/topo/{}_edge.csv'.format(dataset), delimiter=' ')
    # add weight, capacity, delay to edge attributes
    for _, row in df.iterrows():
        i = row.src
        j = row.dest
        G.add_edge(i, j, weight=row.weight,
                   capacity=row.bw,
                   delay=row.delay)
    return G


def draw_network_topology(G, pos=None):
    if pos is None:
        pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_size=1000, alpha=0.5)
    nx.draw_networkx_labels(G, pos)


def shortest_path(G, source, target):
    return nx.shortest_path(G, source=source, target=target, weight='weight')


#####################################
# data structure for heuristic solver
#####################################
def initialize_link2flow(G):
    '''
    link2flow is a dictionary:
        - key: link id (u, v)
        - value: list of flows id (i, j)
    '''
    link2flow = {}
    for u, v in G.edges:
        link2flow[(u, v)] = []
    return link2flow


def is_simple_path(path):
    '''
    input:
        - path: which is a list of edges (u, v)
    return:
        - is_simple_path: bool
    '''
    edges = []
    for edge in path:
        edge = tuple(sorted(edge))
        if edge in edges:
            return False
        edges.append(edge)
    return True


def get_path(G, i, j, k):
    '''
    get a path for flow (i, j) with middle point k
    return:
        - list of edges on path
    '''
    p_ik = shortest_path(G, i, k)
    p_kj = shortest_path(G, k, j)
    edges = []
    # compute edges from path p_ik, p_kj (which is 2 lists of nodes)
    if len(p_ik) > 1:
        for u, v in zip(p_ik[:-1], p_ik[1:]):
            edges.append((u, v))
        for u, v in zip(p_kj[:-1], p_kj[1:]):
            edges.append((u, v))
    return edges


def remove_duplicated_path(paths):
    # convert path to string
    paths = ['-'.join('{}/{}'.format(u, v) for u, v in path) for path in paths]
    # remove duplicated string
    paths = list(set(paths))
    # convert string to path
    new_paths = []
    for path in paths:
        new_path = []
        for edge_str in path.split('-'):
            u, v = edge_str.split('/')
            u, v = int(u), int(v)
            new_path.append((u, v))
        new_paths.append(new_path)
    return new_paths


def sort_paths(G, paths):
    weights = [[sum(G.get_edge_data(u, v)['weight'] for u, v in path)] for path in paths]
    paths = [path for weights, path in sorted(zip(weights, paths), key=lambda x: x[0])]
    return paths


def get_paths(G, i, j):
    '''
    get all simple path for flow (i, j) on graph G
    return:
        - flows: list of paths
        - path: list of links on path (u, v)
    '''
    N = G.number_of_nodes()
    paths = []
    if i != j:
        for k in range(N):
            try:
                path = get_path(G, i, j, k)
                if path:  # if there exists path
                    # in other word, path is not []
                    if is_simple_path(path):
                        paths.append(path)
            except nx.NetworkXNoPath:
                pass
        # remove redundant paths
        paths = remove_duplicated_path(paths)
        # sort paths by their total link weights for heuristic
        paths = sort_paths(G, paths)
    return paths


def initialize_flow2link(G):
    '''
    flow2link is a dictionary:
        - key: flow id (i, j)
        - value: list of paths
        - path: list of links on path (u, v)
    '''
    N = G.number_of_nodes()
    flow2link = {}
    for i, j in itertools.product(range(N), range(N)):
        paths = get_paths(G, i, j)
        flow2link[i, j] = paths
    return flow2link


def get_solution_bound(G, flow2link):
    N = G.number_of_nodes()
    lb = np.zeros([N, N], dtype=int)
    ub = np.empty([N, N], dtype=int)
    for i, j in itertools.product(range(N), range(N)):
        ub[i, j] = len(flow2link[(i, j)])
    ub[ub == 0] = 1
    return lb, ub


def count_routing_change(solution1, solution2):
    return np.sum(solution1 != solution2)
#####################################
