import itertools
from copy import deepcopy

import numpy as np
import pulp as pl

from . import util


class OneStepSRSolver:

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
        self.var_dict = {}
        self.solution = None

    def create_problem(self, tm):
        # save input traffic matrix
        self.tm = tm

        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        segments = self.segments

        # 1) create optimization model
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0.0, cat='Continuous')

        x = pl.LpVariable.dicts(name='x',
                                indexs=np.arange(num_node ** 3),
                                cat='Binary')

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function
        for u, v in G.edges:
            capacity = G.get_edge_data(u, v)['capacity']
            load = pl.lpSum(
                x[util.flatten_index(i, j, k, num_node)] * tm[i, j] * util.g(segments[i][j][k], u, v) for i, j, k in
                itertools.product(range(num_node), range(num_node), range(num_node)))
            problem += load <= theta * capacity

        # 3) constraint function
        # ensure all traffic are routed
        for i, j in itertools.product(range(num_node), range(num_node)):
            problem += pl.lpSum(x[util.flatten_index(i, j, k, num_node)] for k in range(num_node)) >= 1.0

        return problem, x

    def extract_solution(self, problem):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        # extract solution
        for v in problem.variables():
            self.var_dict[v.name] = v.varValue

        self.solution = np.empty([num_node, num_node, num_node])
        for i, j, k in itertools.product(range(num_node), range(num_node), range(num_node)):
            index = util.flatten_index(i, j, k, num_node)
            self.solution[i, j, k] = self.var_dict['x_{}'.format(index)]

    def init_solution(self):
        G = self.G
        num_node = G.number_of_nodes()
        # extract solution
        self.solution = np.zeros([num_node, num_node, num_node])
        for i, j in itertools.product(range(num_node), range(num_node)):
            self.solution[i, j, i] = 1

    def extract_utilization(self, tm):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        segments = self.segments
        # extract utilization
        utilizations = []
        for u, v in G.edges:
            load = sum([self.solution[i, j, k] * util.g(segments[i][j][k], u, v) for i, j, k in
                        itertools.product(range(num_node), range(num_node), range(num_node))])
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            G[u][v]['utilization'] = utilization

    def extract_utilization_v2(self, tm):
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        segments = self.segments
        # recompute the solution, proportional to new demand
        solution = deepcopy(self.solution)

        if solution.any() < 0 or solution.any() > 1 or 1 > solution.any() > 0:
            raise RuntimeError('Infeasible solution')

        # extract utilization
        utilizations = []
        for u, v in G.edges:
            load = sum([solution[i, j, k] * tm[i, j] * util.g(segments[i][j][k], u, v) for i, j, k in
                        itertools.product(range(N), range(N), range(N))])
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            G[u][v]['utilization'] = utilization

    def extract_status(self, problem):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        # extract status
        self.status = pl.LpStatus[problem.status]

    def solve(self, tm):
        # extract parameters
        problem, x = self.create_problem(tm)

        # solver = pl.get_solver('GUROBI', time_limit=60)
        # print(solver)
        # problem.solve(solver)
        self.init_solution()
        problem.solve()
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
