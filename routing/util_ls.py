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


def get_path(G, i, j, k):
    # save calculated path
    p1 = shortest_path(G, i, k)
    p2 = shortest_path(G, k, j)
    path = p1 + p2[1:]
    return tuple(path)


def is_simple_path(path):
    edges = []
    for u, v in zip(path[:-1], path[1:]):
        edge = tuple(sorted(set([u, v])))
        if edge in edges:
            return False
        edges.append(edge)
    return True


def get_paths(G):
    N = G.number_of_nodes()
    paths = [[[] for _ in range(N)] for _ in range(N)]
    for i, j in itertools.product(range(N), range(N)):
        if i != j:
            for k in range(N):
                try:
                    path = get_path(G, i, j, k)
                    flag_simple_path = is_simple_path(path)
                    if flag_simple_path:
                        paths[i][j].append(path)
                except nx.NetworkXNoPath:
                    pass
            paths[i][j] = list(set(paths[i][j]))
    return np.asarray(paths)


def g(path, u, v):
    value = 0
    # count number of occurence of edge (u, v) on path
    for i in range(len(path) - 1):
        if path[i] == u and path[i + 1] == v or \
                path[i] == v and path[i + 1] == u:
            value += 1
    if value >= 2:
        value = 100
    return value


def get_max_link(G):
    utilizations = nx.get_edge_attributes(G, 'utilization')
    u_max, v_max = None, None
    utilization_max = 0
    for u, v in utilizations:
        utilization = utilizations[u, v]
        if utilization > utilization_max:
            u_max, v_max, utilization_max = u, v, utilization
    return u_max, v_max


def get_randomized_max_link(G, alpha=1):
    utilizations_on_graph = nx.get_edge_attributes(G, 'utilization')
    edges = []
    utilizations = []
    for u, v in utilizations_on_graph:
        utilization = utilizations_on_graph[u, v]
        edges.append((u, v))
        utilizations.append(utilization)
    utilizations = np.array(utilizations)
    p = utilizations ** alpha / np.sum(utilizations ** alpha)
    idx = np.random.choice(np.arange(len(edges)), size=1, replace=True, p=p)[0]
    u_max, v_max = edges[idx]
    return u_max, v_max


def get_flows_though_link(solver, solution, u, v):
    G = solver.G
    paths = solver.paths
    N = G.number_of_nodes()
    for i, j in itertools.product(range(N), range(N)):
        if paths[i, j]:
            path = paths[i, j][solution[i, j]]
            if g(path, u, v) > 0:
                yield i, j, path


def get_max_flow_though_link(solver, solution, u, v, tm):
    i_max, j_max, demand_max = 0, 0, 0
    for i, j, path in get_flows_though_link(solver, solution, u, v):
        if tm[i, j] > demand_max:
            i_max, j_max, demand_max = i, j, tm[i, j]
    return i_max, j_max


def get_randomized_max_flow_though_link(solver, solution, u, v, tm, alpha=1):
    flows = []
    demands = []
    for i, j, path in get_flows_though_link(solver, solution, u, v):
        flows.append((i, j))
        demands.append(tm[i, j])
    demands = np.array(demands)
    p = demands ** alpha / np.sum(demands ** alpha)
    idx = np.random.choice(np.arange(len(flows)), size=1, replace=True, p=p)[0]
    i_max, j_max = flows[idx]
    return i_max, j_max


def get_max_flow(solver, solution, tm):
    u, v = get_max_link(solver.G)
    i, j = get_max_flow_though_link(solver, solution, u, v, tm)
    return i, j


def get_randomized_max_flow(solver, solution, tm, alpha=1):
    u, v = get_randomized_max_link(solver.G, alpha=alpha)
    i, j = get_randomized_max_flow_though_link(solver, solution, u, v, tm, alpha=alpha)
    return i, j


def find_middlepoint(G, i, j, path):
    true_k = None
    for idx in range(1, len(path)):
        # try k, to verify if it is the middlepoint of path
        k = path[idx]
        new_path = np.array(get_path(G, i, j, k))
        if len(path) == len(new_path):
            if (path == new_path).all():
                true_k = k
                break
    #    if true_k == None:
    #        print(i, j, true_k, path, new_path)
    return true_k
