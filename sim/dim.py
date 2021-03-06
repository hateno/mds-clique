import logging
import numpy as np
import sim.topics

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

logger = logging.getLogger()
seed = np.random.RandomState(seed=5) # TODO port to ini file

def calc_mds(topic_dist_values):
    dist_matrix_list = sim.topics.dissim(topic_dist_values)
    dist_matrix = check_array(dist_matrix_list)
    mds = MDS(n_jobs=-1, dissimilarity='precomputed', random_state=seed, verbose=1)
    points = mds.fit_transform(dist_matrix)
    n = len(topic_dist_values)
    print('n: %s' % n)
    num_pairs = (n * (n - 1.0)) / 2.0
    print('num pairs: %s' % num_pairs)
    print('mds stress: %s' % (mds.stress_ / num_pairs))
    eucl_matrix = euclidean_distances(points)
    mystress = sim.topics.calc_stress(dist_matrix, eucl_matrix)
    print('my stress: %s' % (mystress / num_pairs))
    return points

def calc_mds_quads(quads_list, topics, clusters):
    for i, quad_list in enumerate(quads_list):
        topics_quad = np.array([topics[point] for point in quad_list])
        clusters_quad = [clusters[point] for point in quad_list]
        print('\ncalculating mds for points %s' % quad_list)
        points = calc_mds(topics_quad)
