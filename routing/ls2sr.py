import itertools
import time
from copy import deepcopy

import numpy as np

from . import util
from . import util_ls


class OneStepLocalSearch2SRSolver:

    def __init__(self, G, segments, time_limit=60, verbose=False):
        # save parameters
        self.G = G
        N = G.number_of_nodes()

        # enumerate all 2SR paths of G
        # NxN matrix contain
        # list of path
        # each path is a tuple of nodes it goes through
        self.paths = util_ls.get_paths(G)

        # NxN matrix contain number of path per flow
        self.ub = np.array([[len(self.paths[i][j]) for j in range(N)] for i in range(N)])
        self.ub[self.ub == 0] = 1
        self.lb = np.zeros_like(self.ub)
        self.segments = segments
        self.alpha = None
        self.time_limit = time_limit
        self.verbose = verbose

    def evaluate(self, solution, tm, verbose=False):
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        paths = self.paths
        # extract utilization
        utilizations = []
        for u, v in G.edges:
            load = 0
            for i, j in itertools.product(range(N), range(N)):
                if paths[i, j]:
                    load += util_ls.g(paths[i, j][solution[i, j]], u, v) * tm[i, j]

            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            G[u][v]['utilization'] = utilization
            if verbose:
                print('evaluate', u, v, load, capacity, utilization)
            utilizations.append(utilization)
        return max(utilizations)

    def sample(self):
        return np.random.randint(low=self.lb, high=self.ub)

    def initialize(self):
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        # initialize empty sample
        sample = np.empty_like(self.lb)

        for i in range(N):
            for j in range(N):
                # get the link with min hop count
                lengths = np.array(len(self.paths[i, j][k]) for k in range(len(self.paths[i, j])))
                sample[i, j] = np.argmin(lengths)
                sample[i, j] = np.argmin(lengths)
        return sample

    def mutate(self, solution, i, j):
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        solution[i, j] = np.random.randint(low=self.lb[i, j], high=self.ub[i, j])
        return solution

    def solve(self, tm, solution=None):  # time_limit (int) seconds
        # initialize and evaluate initial solution
        # if no initial solution given
        if solution is None:
            solution = self.initialize()
        u = self.evaluate(solution, tm)
        if self.verbose:
            print('initial theta={}'.format(u))
        # initialize solver state
        best_solution = deepcopy(solution)
        theta = u
        tic = time.time()

        num_eval = 0
        while time.time() - tic < self.time_limit:
            num_eval += 1
            i, j = util_ls.get_randomized_max_flow(self, solution, tm)  # TODO: change alpha
            self.mutate(solution, i, j)
            u = self.evaluate(solution, tm)
            if u < theta:
                theta = u
                best_solution = deepcopy(solution)
                if self.verbose:
                    print('n={} t={:0.2f} i={} j={} tm={:0.2f} theta={:0.6f}'.format(
                        num_eval, time.time() - tic, i, j, tm[i, j], theta))
        if self.verbose:
            print('n={} t={:0.2f} i={} j={} tm={:0.2f} theta={:0.6f}'.format(
                num_eval, time.time() - tic, i, j, tm[i, j], theta))

        # self.evaluate(best_solution, tm, verbose=True)
        self.solution = self.decode_solution(best_solution)
        return self.solution, best_solution

    def solve_v2(self, tm, solution=None):  # time_limit (int) seconds
        # initialize and evaluate initial solution
        # if no initial solution given
        if solution is None:
            solution = self.initialize()
        u = self.evaluate(solution, tm)
        if self.verbose:
            print('initial theta={}'.format(u))
        # initialize solver state
        best_solution = deepcopy(solution)
        theta = u
        tic = time.time()

        num_eval = 0
        while time.time() - tic < self.time_limit:
            num_eval += 1
            i, j = util_ls.get_randomized_max_flow(self, solution, tm)  # TODO: change alpha
            # backtrack to best solution
            solution = deepcopy(best_solution)
            self.mutate(solution, i, j)
            u = self.evaluate(solution, tm)
            if u < theta:
                theta = u
                best_solution = deepcopy(solution)
            if self.verbose:
                print('n={} t={:0.2f} i={} j={} tm={:0.2f} theta={:0.6f}'.format(
                    num_eval, time.time() - tic, i, j, tm[i, j], theta))
        if self.verbose:
            print('n={} t={:0.2f} i={} j={} tm={:0.2f} theta={:0.6f}'.format(
                num_eval, time.time() - tic, i, j, tm[i, j], theta))

        # self.evaluate(best_solution, tm, verbose=True)
        self.solution = self.decode_solution(best_solution)
        return self.solution, best_solution

    def decode_solution(self, solution):  # TODO Debug
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        paths = self.paths

        alpha = np.zeros([N, N, N])
        for i in range(N):
            for j in range(N):
                if paths[i, j]:
                    path = np.array(paths[i, j][solution[i, j]])
                    k = util_ls.find_middlepoint(G, i, j, path)
                    if k is not None:
                        alpha[i, j, k] = 1
                        segment = self.segments[i][j][k]
        return alpha

    def extract_utilization_v2(self, tm):
        # extract parameters
        G = self.G
        N = G.number_of_nodes()
        segments = self.segments
        # recompute the solution, proportional to new demand
        solution = deepcopy(self.solution)

        if solution.any() < 0 or solution.any() > 1 or 1 > solution.any() > 0:
            raise RuntimeError('Infeasible solution')

        for u, v in G.edges:
            load = sum([solution[i, j, k] * tm[i, j] * util.g(segments[i][j][k], u, v) for i, j, k in
                        itertools.product(range(N), range(N), range(N))])
            capacity = G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            # print('extract_utilization_v2', u, v, load, capacity, utilization)
            G[u][v]['utilization'] = utilization
