import argparse, itertools, logging, math, os, pickle, random, shutil, sys, tvconf
import numpy as np
import sim.dim, sim.graph, sim.random, sim.relative, sim.clique, sim.online, sim.output, sim.topics, sim.util

from multiprocessing import Manager, Pool
from multiprocessing.pool import ThreadPool
from sim.mycorpus import MyCorpus
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Perform different functions')
parser.add_argument('-dim', type=int, required=True, help='Dimension of MDS output')
parser.add_argument('-data', choices=['corpus', 'random', 'none'], default='random', required=True, help='Specify input data to use')
parser.add_argument('-n', type=int, default=100, help='Number of randomly generated topics, used in conjunction with -data=random')
parser.add_argument('-ndim', type=int, default=3, help='Used with -data=random, the dimension of the randomly generated points')
parser.add_argument('-clusters', type=int, default=3, help='Number of clusters to extract')
parser.add_argument('-clique', type=str, default='stress', help='Use distance or standard deviation measure')
parser.add_argument('--approximation', action='store_true', help='Use approximate clique algorithm')
parser.add_argument('--parallel', action='store_true', help='Use parallel processing whereever possible')
parser.add_argument('--matrix', action='store_true', help='Use a dist matrix instead of topic points, replaces topic_dist_values with dummy points')

parser.add_argument('-ident', type=str, default='', help='Store corpus and any other data in this path under this particular ident')

# EXPERIMENTS
parser.add_argument('-e', type=int, default=8, help='Number of samples to run for an experiment')
parser.add_argument('-cores', type=int, default=os.cpu_count() // 4, help='Number of cores to run for an experiment')
parser.add_argument('--relative', action='store_true', help='Perform relative MDS calculation')
parser.add_argument('--rclique', action='store_true', help='Perform clique graph/relative MDS calculation')
parser.add_argument('--online', action='store_true', help='Perform online (one-by-one) relative MDS calculation')
parser.add_argument('--onlineclique', action='store_true', help='Perform online clique experiment')
parser.add_argument('--relativeonline', action='store_true', help='Perform Online Relative MDS clique experiment')
parser.add_argument('--single', action='store_true', help='Run experiment in single process mode (useful for debugging)')
parser.add_argument('--combine', action='store_true', help='Alternative clique graph algorithm that combines max cliques')
parser.add_argument('--rmds', action='store_true', help='Run --rclique with Relative MDS calculation')

args = parser.parse_args()

sim.util.preprocessing(args.ident) # preprocess

### RETRIEVE DATA ###
if args.data != 'none':
    if args.data == 'corpus':
        # ap topics
        (topic_dist, topic_dist_values) = sim.topics.load_topics()
        dist_matrix_list = sim.topics.dissim(topic_dist_values, ident=args.ident)
        dist_matrix = check_array(dist_matrix_list)

        # clusters of mds
        clusters = sim.topics.cluster_topics(args.clusters, topic_dist_values)
        cluster_scores = sim.topics.cluster_validation(topic_dist_values, clusters)

    elif args.data == 'random':
        topic_dist_values, dist_matrix, clusters = sim.util.generate_data(n=args.n, ndim=args.ndim, n_clusters=args.clusters, ident=args.ident, matrix=args.matrix)

### RUN MDS ###
if args.relative: # relative mds
    logger.info('Running Relative MDS...')
    n_samples = args.e
    K_MIN = 5
    K_MAX = 100
    K_STEP = 5
    N_PROCESS = args.cores
    rand = random.SystemRandom()

    all_f = []
    manager = Manager()
    for k in range(K_MIN, K_MAX, K_STEP):
        # store various stress values per file for this k
        stress = manager.dict()
        samples = []
        population = list(range(100))
        subdir = '%d-all' % k

        for _ in range(n_samples):
            sample = rand.sample(population, k)
            samples.append(sample)

        sample_ids = list(enumerate(samples))
        pargs = [(sample_id[0], sample_id[1], stress, k, args.dim, dist_matrix, clusters) for sample_id in sample_ids]
        pool = Pool(N_PROCESS)
        pool.starmap(sim.util.process_stress, pargs)
        pool.close()
        pool.join()

        fstress = np.array(list(stress.values()))
        flist = sorted(list(stress.items()), key=lambda tupl: tupl[1])

        fmin = flist[0][0]
        fmedian = flist[len(flist) // 2][0]
        fmax = flist[-1][0]
        print('final_stress: %f %f %f %f' % (fstress.mean(), fstress.std(), fstress.min(), fstress.max()))
        print('\t(min, median, max): %s %s %s' % (fmin, fmedian, fmax))

        path = 'out/experiment/%s' % subdir
        final_path = 'out/final/%s' % k
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        shutil.copy('%s/%s_relative' % (path, fmin), '%s/amin_relative' % final_path)
        shutil.copy('%s/%s_relative' % (path, fmedian), '%s/amedian_relative' % final_path)
        shutil.copy('%s/%s_relative' % (path, fmax), '%s/amax_relative' % final_path)

        all_f.append((fstress.min(), fstress.mean(), fstress.max()))

elif args.rmds: # rmds algorithm
    logger.info('Running RMDS...')
    N_EXPERIMENTS = args.e
    N_PROCESS = args.cores

    manager = Manager()
    final_result = manager.dict()

    if args.single:
        logger.info('Running rmds in single mode...')
        sim.util.process_rmds(0, args, final_result)
    else:
        logger.info('Running rmds in parallel mode...')
        experiment_ids = list(range(N_EXPERIMENTS))
        pargs = [(experiment_id, args, final_result) for experiment_id in experiment_ids]
        pool = Pool(N_PROCESS)
        pool.starmap(sim.util.process_rmds, pargs)
        pool.close()
        pool.join()

    sim.util.analyze_rmds(final_result, args, N_EXPERIMENTS)

elif args.rclique: # clique algorithm
    logger.info('Running Clique...')
    N_EXPERIMENTS = args.e
    N_PROCESS = args.cores

    manager = Manager()
    final_result = manager.dict()
    if args.single:
        logger.info('Running rclique in single mode...')
        sim.util.process_rclique(0, args, final_result)
    else:
        logger.info('Running rclique in parallel mode...')
        experiment_ids = list(range(N_EXPERIMENTS))
        pargs = [(experiment_id, args, final_result) for experiment_id in experiment_ids]
        pool = Pool(N_PROCESS)
        pool.starmap(sim.util.process_rclique, pargs)
        pool.close()
        pool.join()

    sim.util.analyze_rclique(final_result, args, N_EXPERIMENTS)

elif args.relativeonline: # online relative mds clique algorithm
    logger.info('Running Relative Online Clique...')
    N_EXPERIMENTS = args.e
    N_PROCESS = args.cores

    manager = Manager()
    final_result = manager.dict()
    if args.single:
        sim.util.process_relative_online_clique(0, args, final_result)
    else:
        experiment_ids = list(range(N_EXPERIMENTS))
        pargs = [(experiment_id, args, final_result) for experiment_id in experiment_ids]
        pool = Pool(N_PROCESS)
        pool.starmap(sim.util.process_relative_online_clique, pargs)
        pool.close()
        pool.join()

    sim.util.analyze_relativeonline(final_result, args, N_EXPERIMENTS)

elif args.onlineclique: # online clique algorithm
    logger.info('Running Online Clique...')
    N_EXPERIMENTS = args.e
    N_PROCESS = args.cores

    manager = Manager()
    final_result = manager.dict()
    if args.single:
        sim.util.process_online_clique(0, args, final_result)
    else:
        experiment_ids = list(range(N_EXPERIMENTS))
        pargs = [(experiment_id, args, final_result) for experiment_id in experiment_ids]
        pool = Pool(N_PROCESS)
        pool.starmap(sim.util.process_online_clique, pargs)
        pool.close()
        pool.join()

    sim.util.analyze_onlineclique(final_result, args, N_EXPERIMENTS)

elif args.online: # data coming in one by one
    logger.info('Running Online...')
    N_SAMPLES = args.e
    N_PROCESS = args.cores

    manager = Manager()
    all_stress = manager.dict()
    all_stress['controls'] = []
    if args.single:
        sim.util.process_online(0, args, all_stress)
    else:
        sample_ids = list(range(N_SAMPLES))
        pargs = [(sample_id, args, all_stress) for sample_id in sample_ids]
        pool = Pool(N_PROCESS)
        pool.starmap(sim.util.process_online, pargs)
        pool.close()
        pool.join()

    sim.util.analyze_online(all_stress)

else: # use scikit's mds
    logger.info('Running MDS...')
    # mds points calculation
    points = sim.topics.mds_helper(dist_matrix, rdim=args.dim)
    eucl_matrix = euclidean_distances(points)

    sim.topics.print_mds_points(points, clusters=clusters)

    # cluster of points
    clusters_p = sim.topics.cluster_topics(args.clusters, points)
    cluster_scores_p = sim.topics.cluster_validation(points, clusters_p)

    # mds graph initialize
    mgraph = sim.graph.MGraph(points, topic_dist_values, dist_matrix, eucl_matrix, args.dim, clusters, parallel=args.parallel)
    if args.clique == 'stress':
        cliques = mgraph.find_cliques_stress(combine=args.combine)
    elif args.clique == 'distance':
        cliques = mgraph.find_cliques_distance(0.1, combine=args.combine)
    mgraph.write_to_file()
