import itertools
from copy import deepcopy

import numpy as np
import pulp as pl

from . import util


class P1Solver:

    def __init__(self, G, segments):
        '''
        G: networkx Digraph, a network topology
        '''
        self.G = G
        # self.segments = util.get_segments(G)
        self.segments = segments
        self.tm = None
        self.num_tms = 0
        self.problem = None
        self.nflows = 0
        self.nnodes = 0
        self.solution = None
        self.status = None

    def flatten_index(self, i, j):
        return i * self.nnodes + j

    def create_problem(self, tm):
        # save input traffic matrix
        self.tm = tm
        assert len(self.tm.shape) == 4

        t, n, n, f = tm.shape

        self.num_tms = t
        self.nflows = f

        # extract parameters
        G = self.G
        self.nnodes = G.number_of_nodes()
        assert self.nnodes == n
        segments = self.segments

        # 1) create optimization model
        problem = pl.LpProblem('SP', pl.LpMinimize)
        # theta = pl.LpVariable(name='theta', lowBound=0.0, cat='Continuous')
        x = pl.LpVariable.dicts(name='x',
                                indexs=np.arange(self.nnodes * self.nflows),
                                cat='Binary')

        # 2) objective function
        # minimize maximum link utilization
        problem += pl.lpSum(x[self.flatten_index(k, f)] * tm[] for k, f in itertools.product(range(self.nnodes),
                                                                                             range(self.nflows)))

        # 3) constraint function
        for u, v in G.edges:
            capacity = G.get_edge_data(u, v)['capacity']
            for t in range(self.num_tms):
                load = pl.lpSum(
                    x[self.flatten_index(i, j, k, f)] * tm[t, i, j, f] * util.g(segments[i][j][k], u, v)
                    for i, j, k, f in
                    itertools.product(range(self.nnodes), range(self.nnodes), range(self.nnodes), range(self.nflows)))
                problem += load <= theta * capacity

        # 3) constraint function
        # ensure all traffic are routed
        for i, j, f in itertools.product(range(self.nnodes), range(self.nnodes), range(self.nflows)):
            problem += pl.lpSum(x[self.flatten_index(i, j, k, f)] for k in range(self.nnodes)) == 1.0

        return problem, x

    def extract_solution(self, problem):
        d = {}
        for v in problem.variables():
            d[v.name] = v.varValue

        self.solution = np.empty([self.nnodes, self.nnodes, self.nnodes, self.nflows])
        for i, j, k, f in itertools.product(range(self.nnodes), range(self.nnodes), range(self.nnodes),
                                            range(self.nflows)):
            index = self.flatten_index(i, j, k, f)
            self.solution[i, j, k, f] = d['x_{}'.format(index)]

    def extract_utilization(self, tm):
        # extract parameters
        segments = self.segments
        # extract utilization
        for u, v in self.G.edges:
            load = sum([self.solution[i, j, k, f] * tm[i, j, f] * util.g(segments[i][j][k], u, v) for i, j, k, f in
                        itertools.product(range(self.nnodes), range(self.nnodes), range(self.nnodes),
                                          range(self.nflows))])
            capacity = self.G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            self.G[u][v]['utilization'] = utilization

    def extract_utilization_v2(self, tm):
        # extract parameters
        # recompute the solution, proportional to new demand

        assert len(tm.shape) == 3
        segments = self.segments
        solution = deepcopy(self.solution)

        if solution.any() < 0 or solution.any() > 1 or 1 > solution.any() > 0:
            raise RuntimeError('Infeasible solution')

        # extract utilization
        for u, v in self.G.edges:
            load = sum([solution[i, j, k, f] * tm[i, j, f] * util.g(segments[i][j][k], u, v) for i, j, k, f in
                        itertools.product(range(self.nnodes), range(self.nnodes), range(self.nnodes),
                                          range(self.nflows))])
            capacity = self.G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            self.G[u][v]['utilization'] = utilization

    def extract_status(self, problem):
        # extract parameters
        self.status = pl.LpStatus[problem.status]

    def solve(self, tms):
        # extract parameters
        problem, x = self.create_problem(tms)
        # _s = time.time()
        problem.solve()
        # print('Solving time:', time.time() - _s)
        self.problem = problem

        self.extract_status(problem)
        self.extract_solution(problem)

    def get_paths(self, i, j):
        G = self.G
        if i == j:
            list_k = [i]
        else:
            list_k = np.where(self.solution[i, j] > 0)[0]
        paths = []
        for k in list_k:
            path = []
            path += util.shortest_path(G, i, k)[:-1]
            path += util.shortest_path(G, k, j)
            paths.append((k, path))
        return paths
