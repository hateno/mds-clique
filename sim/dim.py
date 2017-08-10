import logging
import matplotlib.pyplot as plt
import numpy as np
import sim.topics, sim.graphit

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
seed = np.random.RandomState(seed=5) # TODO port to ini file

def calc_mds(topic_dist_values):
    dist_matrix_list = sim.topics.dissim(topic_dist_values)
    dist_matrix = check_array(dist_matrix_list)
    mds = MDS(n_jobs=-1, dissimilarity='precomputed', random_state=seed, verbose=1)
    points = mds.fit_transform(dist_matrix)
    print('mds stress: %s' % mds.stress_)
    eucl_matrix = euclidean_distances(points)
    mystress = sim.topics.calc_stress(dist_matrix, eucl_matrix)
    print('my stress: %s' % mystress)
    return points

def calc_mds_quads(quads_list, topics, clusters):
    for i, quad_list in enumerate(quads_list):
        topics_quad = np.array([topics[point] for point in quad_list])
        clusters_quad = [clusters[point] for point in quad_list]
        print('\ncalculating mds for points %s' % quad_list)
        points = calc_mds(topics_quad)
        sim.graphit.plot(points, clusters_quad, point_clusters=False, title='Quad %i' % i, xlabel='x1', ylabel='x2')
