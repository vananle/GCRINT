import itertools
from copy import deepcopy

import numpy as np
import pulp as pl

from . import util


class ObliviousRoutingSolver:

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

    def flatten_index(self, i, j, num_edge):
        return i * num_edge + j

    def create_problem(self):
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        E = G.number_of_edges()
        segments = self.segments

        # 0) initialize lookup dictionary from index i to edge u, v
        edges_dictionary = {}
        for i, (u, v) in enumerate(G.edges):
            edges_dictionary[i] = (u, v)

        # 1) create optimization model of dual problem
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0.0, cat='Continuous')
        x = pl.LpVariable.dicts(name='x', indexs=np.arange(N ** 3), cat='Binary')
        pi = pl.LpVariable.dicts(name='pi', indexs=np.arange(E ** 2), lowBound=0.0)

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function 2
        for i, j in itertools.product(range(N), range(N)):  # forall ij

            for e_prime in edges_dictionary:  # forall e' = [u, v]
                u, v = edges_dictionary[e_prime]
                # sum(g_ijk(e'))*alpha_ijk
                lb = pl.lpSum(
                    [util.g(segments[i][j][k], u, v) * x[util.flatten_index(i, j, k, N)] for k in range(N)])
                for m in range(N):  # forall m
                    # sum(g_ijm(e) * pi(e,e)') >= sum(g_ijk(e')) * alpha_ijk
                    problem += pl.lpSum([util.g(segments[i][j][m], edges_dictionary[e][0], edges_dictionary[e][1])
                                         * pi[self.flatten_index(e, e_prime, E)] for e in edges_dictionary]) >= lb

        # 4) constraint function 3
        for e_prime in edges_dictionary:  # for edge e'   sum(c(e) * pi(e, e')) <= theta * c(e')
            u, v = edges_dictionary[e_prime]
            capacity_e_prime = G.get_edge_data(u, v)['capacity']
            problem += pl.lpSum([G.get_edge_data(edges_dictionary[e][0], edges_dictionary[e][1])['capacity'] *
                                 pi[self.flatten_index(e, e_prime, E)] for e in edges_dictionary]) \
                       <= theta * capacity_e_prime

        # 3) constraint function 4
        for i, j in itertools.product(range(N), range(N)):  # forall ij:   sunm(alpha_ijk) == 1.0
            problem += pl.lpSum(x[util.flatten_index(i, j, k, N)] for k in range(N)) == 1.0

        return problem

    def init_solution(self):
        G = self.G
        num_node = G.number_of_nodes()
        # extract solution
        self.solution = np.zeros([num_node, num_node, num_node])
        for i, j in itertools.product(range(num_node), range(num_node)):
            self.solution[i, j, i] = 1

    def solve(self):
        print('OR: Solver')

        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        print('     + OR: Creating problem')
        problem = self.create_problem()
        print('     + OR: Solving problem')
        self.init_solution()
        problem.solve()
        self.problem = problem
        print('     + OR: extract_status')
        self.extract_status(problem)
        print('     + OR: extract_solution')
        self.extract_solution(problem)

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

    def extract_status(self, problem):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        # extract status
        self.status = pl.LpStatus[problem.status]

    def extract_utilization_v2(self, tm):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        segments = self.segments
        # recompute the solution, proportional to new demand
        solution = deepcopy(self.solution)

        if solution.any() < 0 or solution.any() > 1 or 1 > solution.any() > 0:
            raise RuntimeError('Infeasible solution')

        # extract utilization
        for u, v in G.edges:
            load = sum([solution[i, j, k] * tm[i, j] * util.g(segments[i][j][k], u, v) for i, j, k in
                        itertools.product(range(num_node), range(num_node), range(num_node))])
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            G[u][v]['utilization'] = utilization
