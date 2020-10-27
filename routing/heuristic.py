import itertools
import time

import networkx as nx
import numpy as np

from . import util_h as uh


class HeuristicSolver:

    def __init__(self, G, time_limit=10, verbose=False):
        # save parameters
        self.G = G
        N = G.number_of_nodes()
        self.time_limit = time_limit
        self.verbose = verbose

        # compute paths
        self.link2flow = uh.initialize_link2flow(G)
        self.flow2link = uh.initialize_flow2link(G)
        self.lb, self.ub = uh.get_solution_bound(G, self.flow2link)

        # data for selecting next link -> demand to be mutate
        self.link_selection_prob = None
        self.demand_selection_prob = None

        # cache
        self.tm = None

    def initialize(self):
        return np.zeros_like(self.lb)

    def g(self, i, j, u, v, k):
        if (u, v) in self.flow2link[(i, j)][k] or \
                (v, u) in self.flow2link[(i, j)][k]:
            return 1
        return 0

    def has_path(self, i, j):
        if self.flow2link[(i, j)]:
            return True
        return False

    def set_link_selection_prob(self, alpha=16):
        # extract parameters
        G = self.G
        # compute the prob
        utilizations = nx.get_edge_attributes(G, 'utilization').values()
        utilizations = np.array(list(utilizations))
        self.link_selection_prob = utilizations ** alpha / np.sum(utilizations ** alpha)

    def set_flow_selection_prob(self, alpha=1):
        # extract parameters
        G = self.G
        tm = self.tm
        # compute the prob
        self.demand_selection_prob = {}
        for u, v in G.edges:
            demands = np.array([tm[i, j] for i, j in self.link2flow[(u, v)]])
            self.demand_selection_prob[(u, v)] = demands ** alpha / np.sum(demands ** alpha)

    def select_flow(self):
        # extract parameters
        G = self.G
        # select link
        indices = np.arange(len(G.edges))
        index = np.random.choice(indices, p=self.link_selection_prob)
        link = list(G.edges)[index]
        # select flow
        indices = np.arange(len(self.link2flow[link]))
        index = np.random.choice(indices, p=self.demand_selection_prob[link])
        flow = self.link2flow[link][index]
        return flow

    def edge_in_path(self, edge, path):
        '''
        input:
            - edge: tuple (u, v)
            - path: list of tuple (u, v)
        '''
        sorted_edge = tuple(sorted(edge))
        sorted_path_edges = [tuple(sorted(path_edge)) for path_edge in path]
        if edge in sorted_path_edges:
            return True
        return False

    def set_link2flow(self, solution):
        # extract parameters
        G = self.G
        # initialize link2flow
        self.link2flow = {}
        for edge in G.edges:
            self.link2flow[edge] = []
        # compute link2flow
        for edge in G.edges:
            for i, j in self.flow2link:
                k = solution[i, j]
                if self.has_path(i, j):
                    path = self.flow2link[i, j][k]
                    if self.edge_in_path(edge, path):
                        self.link2flow[(edge)].append((i, j))

    def set_lowerbound(self, solution):
        self.lb = solution.copy()

    def set_G(self, G):
        self.G = G

    def evaluate(self, solution, tm=None, save_utilization=False):
        # extract parameters
        if save_utilization:
            G = self.G
        else:
            G = self.G.copy()
        N = G.number_of_nodes()
        if tm is None:
            tm = self.tm
        # extract utilization
        utilizations = []
        for u, v in G.edges:
            load = 0
            demands = []
            for i, j in itertools.product(range(N), range(N)):
                if self.has_path(i, j):
                    k = solution[i, j]
                    load += self.g(i, j, u, v, k) * tm[i, j]
                    if self.g(i, j, u, v, k):
                        demands.append((i, j))
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            G[u][v]['utilization'] = utilization
            utilizations.append(utilization)
        return max(utilizations)

    def mutate(self, solution, i, j):
        self.lb[i, j] = self.lb[i, j] + 1
        if self.lb[i, j] >= self.ub[i, j]:
            self.lb[i, j] = 0
        solution[i, j] = self.lb[i, j]
        return solution

    def evaluate_fast(self, solution, best_solution, i, j):
        # extract parameters
        G = self.G.copy()
        tm = self.tm

        # extract old and new path
        k = solution[i, j]
        best_k = best_solution[i, j]
        path = self.flow2link[(i, j)][k]
        best_path = self.flow2link[(i, j)][best_k]

        # accumulate the utilization
        for u, v in best_path:
            u, v = sorted([u, v])
            G[u][v]['utilization'] -= tm[i, j] / G[u][v]['capacity']
        for u, v in path:
            u, v = sorted([u, v])
            G[u][v]['utilization'] += tm[i, j] / G[u][v]['capacity']
        # get all utilizations from edges
        utilizations = nx.get_edge_attributes(G, 'utilization').values()
        return max(utilizations), G

    def solve(self, tm, solution=None, eps=1e-3):
        # save parameters
        self.tm = tm

        # initialize solution
        if solution is None:
            solution = self.initialize()

        # initialize solver state
        self.set_link2flow(solution)
        best_solution = solution.copy()
        u = self.evaluate(solution, save_utilization=True)
        theta = u
        self.set_link2flow(best_solution)
        self.set_link_selection_prob()
        self.set_flow_selection_prob()
        self.set_lowerbound(best_solution)
        tic = time.time()

        if self.verbose:
            print('initial theta={}'.format(u))

        # iteratively solve
        num_eval = 0
        while time.time() - tic < self.time_limit:
            num_eval += 1
            i, j = self.select_flow()
            solution = best_solution.copy()
            solution = self.mutate(solution, i, j)
            u, G = self.evaluate_fast(solution, best_solution, i, j)
            # u_exact = self.evaluate(solution)
            # np.testing.assert_almost_equal(u, u_exact, decimal=6)
            if u - theta < -eps:
                best_solution = solution.copy()
                u_exact = self.evaluate(best_solution, save_utilization=True)
                np.testing.assert_almost_equal(u, u_exact, decimal=6)
                theta = u_exact
                self.set_link2flow(best_solution)
                self.set_link_selection_prob()
                self.set_flow_selection_prob()
                self.set_lowerbound(best_solution)
                self.set_G(G)
                if self.verbose:
                    print('[+] new solution found n={} t={:0.2f} i={} j={} tm={:0.2f} theta={}'.format(
                        num_eval, time.time() - tic, i, j, tm[i, j], theta))
        if self.verbose:
            print('[+] final solution: n={} t={:0.2f} i={} j={} tm={:0.2f} theta={:0.6f}'.format(
                num_eval, time.time() - tic, i, j, tm[i, j], theta))
        return best_solution
