import os

from .one_step_sr import OneStepSRSolver
from .util import *
from .util_h import count_routing_change


# G = load_network_topology('abilene_tm')
# segments = get_segments(G)


def get_route_changes(routings, G):
    route_changes = np.zeros(shape=(routings.shape[0] - 1))
    for t in range(routings.shape[0] - 1):
        _route_changes = 0
        for i, j in itertools.product(range(routings.shape[1]), range(routings.shape[2])):
            path_t_1 = get_paths(G, routings[t + 1], i, j)
            path_t = get_paths(G, routings[t], i, j)
            if path_t_1 != path_t:
                _route_changes += 1

        route_changes[t] = _route_changes

    return route_changes


def get_route_changes_heuristic(routings):
    route_changes = []
    for t in range(routings.shape[0] - 1):
        route_changes.append(count_routing_change(routings[t + 1], routings[t]))

    route_changes = np.asarray(route_changes)
    return route_changes


def extract_results(results):
    mlus, solutions = [], []
    for _mlu, _solution in results:
        mlus.append(_mlu)
        solutions.append(_solution)

    mlus = np.stack(mlus, axis=0)
    solutions = np.stack(solutions, axis=0)

    return mlus, solutions


def save_results(log_dir, fname, mlus, route_change):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    np.save(os.path.join(log_dir, fname + '_mlus'), mlus)
    np.save(os.path.join(log_dir, fname + '_route_change'), route_change)


def get_te_data(x_gt, y_gt, yhat, args):
    te_step = args.test_size if args.te_step is 0 else args.te_step
    nsteps = len(range(0, te_step, args.seq_len_y))

    if x_gt.shape[0] > nsteps * 2:
        x_gt = x_gt[0:te_step:args.seq_len_y]
        y_gt = y_gt[0:te_step:args.seq_len_y]
        yhat = yhat[0:te_step:args.seq_len_y]

    return x_gt, y_gt, yhat


def run_te(x_gt, y_gt, yhat, args):
    if 'abilene' in args.dataset:
        dataset = 'abilene_tm'
    elif 'brain' in args.dataset:
        dataset = 'brain_tm'
    elif 'geant' in args.dataset:
        dataset = 'geant_tm'
    else:
        raise NotImplementedError

    G = load_network_topology(dataset)

    if not os.path.isfile('../../../data/topo/{}_segments.npy'.format(dataset)):

        segments = get_segments(G)
        np.save('../../../data/topo/{}_segments'.format(dataset), segments)
    else:
        segments = np.load('../../../data/topo/{}_segments.npy'.format(dataset), allow_pickle=True)

    x_gt, y_gt, yhat = get_te_data(x_gt, y_gt, yhat, args)

    te_step = x_gt.shape[0]

    print('    Method           |   Min     Avg    Max     std')

    results_optimal = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
        c='optimal', tms=yhat[i], gt_tms=y_gt[i], G=G, nNodes=args.nNodes) for i in range(te_step))

    mlu_optimal, solution_optimal = extract_results(results_optimal)
    solution_optimal = np.reshape(solution_optimal, newshape=(-1, args.nNodes, args.nNodes, args.nNodes))
    route_changes_opt = get_route_changes(solution_optimal, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_opt), np.std(route_changes_opt)))
    print('Optimal              | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu_optimal),
                                                                            np.mean(mlu_optimal),
                                                                            np.max(mlu_optimal),
                                                                            np.std(mlu_optimal)))
    save_results(args.log_dir, 'optimal_optimal', mlu_optimal, route_changes_opt)

    # results_optimal = Parallel(n_jobs=os.cpu_count() - 4)(delayed(do_te)(
    #     c='optimal', tms=y_gt[i], gt_tms=y_gt[i], G=G, nNodes=args.nNodes) for i in range(te_step))
    #
    # mlu_optimal, solution_optimal = extract_results(results_optimal)
    # solution_optimal = np.reshape(solution_optimal, newshape=(-1, args.nNodes, args.nNodes, args.nNodes))
    # route_changes_opt = get_route_changes(solution_optimal, G)
    # print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(route_changes_opt), np.std(route_changes_opt)))
    # print('Optimal              | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu_optimal),
    #                                                                         np.mean(mlu_optimal),
    #                                                                         np.max(mlu_optimal),
    #                                                                         np.std(mlu_optimal)))
    # save_results(args.log_dir, 'optimal', mlu_optimal, route_changes_opt)


def do_te(c, tms, gt_tms, G, nNodes=12):
    tms = tms.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))

    segments = get_segments(G)

    if c == 'optimal':
        optimal_solver = OneStepSRSolver(G, segments)
        return optimal_sr(optimal_solver, gt_tms, tms)
    else:
        raise ValueError('TE not found')


def optimal_sr(solver, gt_tms, tms):
    u = []
    solutions = []
    for i in range(gt_tms.shape[0]):
        try:
            solver.solve(tms[i])
        except:
            pass
        solutions.append(solver.solution)
        u.append(get_max_utilization_v2(solver, gt_tms[i]))

    solutions = np.stack(solutions, axis=0)
    return u, solutions
