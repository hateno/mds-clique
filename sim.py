import itertools, logging, os, json, tvconf, argparse, math, sys, random
import networkx as nx
import numpy as np
import sim.dim, sim.graph, sim.random, sim.relative, sim.output, sim.topics

from sim.mycorpus import MyCorpus
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Perform different functions')
parser.add_argument('-dim', type=int, required=True, help='Dimension of MDS algorithm')
parser.add_argument('-corpus', choices=['ap', 'random', 'random-dist'], default='ap', required=True, help='Type of corpus to use [ap, random, random-dist]')
parser.add_argument('-n', type=int, default=100, help='Number of randomly generated topics, used in conjunction with -corpus=random')
parser.add_argument('-rdim', type=int, default=3, help='Used with -corpus=random, the dimension of the randomly generated points')
parser.add_argument('-clusters', type=int, default=10, help='Number of clusters to extract')
parser.add_argument('-clique', choices=['distance', 'stress'], default='distance', help='Criteria to remove edges of a mesh graph')
parser.add_argument('-k', type=int, default=0.1, help='Only used when -clique=distance, k% at which to remove distance')
parser.add_argument('--flush', action='store_true', help='Clear pickle cache, use with caution')
parser.add_argument('--regen', action='store_true', help='Used with -corpus=random, regenerate random points or use previous')
parser.add_argument('--pickle', action='store_false', help='Pickle enabled, store data being generated')
parser.add_argument('--repickle', action='store_false', help='Repickle a data structure, typically used in conjunction with --pickle')
parser.add_argument('--relative', action='store_true', help='Perform relative MDS calculation')
args = parser.parse_args()

# clear out all pickle storage
if args.flush:
    sim.topics.flush()

r_dim = args.dim # decide MDS type
NUM_CLUSTERS = args.clusters # number of clusters to detect

if args.corpus == 'ap' or args.corpus == 'random':
    if args.corpus == 'ap':
        # ap topics
        (topic_dist, topic_dist_values) = sim.topics.load_topics()
        dist_matrix_list = sim.topics.dissim(topic_dist_values, repickle=args.repickle)
        dist_matrix = check_array(dist_matrix_list)

    elif args.corpus == 'random':
        # random
        n = args.corpus[1]
        rand = sim.random.Random(n=args.n)

        # random dist values
        topic_dist_values = rand.generate_cluster_value_matrix(regen=args.regen, rdim=args.rdim)
        topic_dist_values = np.array(topic_dist_values)

        # dist matrix calculation
        dist_matrix_list = sim.topics.dissim(topic_dist_values, pickle_enabled=args.pickle)
        dist_matrix = check_array(dist_matrix_list)

    # clusters of mds
    clusters = sim.topics.cluster_topics(NUM_CLUSTERS, topic_dist_values)
    cluster_scores = sim.topics.cluster_validation(topic_dist_values, clusters)

elif args.corpus == 'random-dist': # random dist matrix
    (clusters, dist_matrix_list) = rand.generate_cluster_dist_matrix()
    dist_matrix = check_array(dist_matrix_list)
    sim.topics.print_mds_points(points, clusters=clusters)

if args.relative: # relative mds
    stress = {}
    for k in range(5,6): # TODO hard-coded the number of basis vectors
        combos = []
        population = list(range(100))
        for _ in range(100):
            combo = random.sample(population, k)
            combos.append(combo)
        for combo_id, basis_ids in enumerate(combos):
            filename = '%s-%s' % (k, combo_id)
            print('processing %s...' % filename)
            relative = sim.relative.Relative(dist_matrix, clusters, r_dim=r_dim, basis=basis_ids)
            relative.print_basis_points(filename=filename)
            relative.print_result(filename=filename)
            stress[filename] = relative.rmds.stress_
else: # use scikit's mds
    # mds points calculation
    points = sim.topics.mds_helper(dist_matrix, r=r_dim, pickle_enabled=args.pickle)
    eucl_matrix = euclidean_distances(points)

    sim.topics.print_mds_points(points, clusters=clusters)

    # cluster of points
    clusters_p = sim.topics.cluster_topics(NUM_CLUSTERS, points)
    cluster_scores_p = sim.topics.cluster_validation(points, clusters_p)

    # calculate stress between every point pairs
    stress_points = sim.topics.list_stress_points(dist_matrix, eucl_matrix, pickle_enabled=False)

    # mds graph initialize
    mgraph = sim.graph.MGraph(points, stress_points, topic_dist_values, dist_matrix, eucl_matrix, r_dim, clusters)

    # mds graph algorithm type
    if args.clique == 'distance':
        cliques = mgraph.find_cliques_distance(args.k)
    elif args.clique == 'stress':
        cliques = mgraph.find_cliques_stress()

    # output to file
    mgraph.write_to_file()
