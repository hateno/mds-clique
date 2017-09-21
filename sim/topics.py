import ctypes as c
import numpy as np
import multiprocessing as mp
import math, os, pickle, shutil, tvconf

from functools import partial
from gensim import models
from sim.distance import Distance
from sim.jqmcvi import dunn_fast
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, calinski_harabaz_score

def flush():
    '''
    Clear all pickle cache
    '''
    print('deleting tmp/flush...')
    shutil.rmtree('tmp/flush', ignore_errors=True)

def pickle_load(filename):
    if not os.path.exists('tmp/flush'):
        os.makedirs('tmp/flush')
    obj = None
    filepath = 'tmp/flush/%s' % filename
    if os.path.exists(filepath):
        print('unpickling %s...' % filename)
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
    else:
        print('pickle %s not found, calculating...' % filename)
    return obj

def pickle_store(filename, obj):
    filepath = 'tmp/flush/%s' % filename
    with open(filepath, 'wb') as f:
        print('pickling %s...' % filename)
        pickle.dump(obj, f)

def load_topics():
    topic_dist_pickle = 'tmp/topic_dist'
    topic_dist_values_pickle = 'tmp/topic_dist_values'
    if os.path.isfile(topic_dist_pickle) and os.path.isfile(topic_dist_values_pickle):
        print('Loading from pickle...')
        with open(topic_dist_pickle, 'rb') as f:
            topic_dist = pickle.load(f)
        with open(topic_dist_values_pickle, 'rb') as f:
            topic_dist_values = pickle.load(f)
    else:
        print('Loading from store...')
        lda = models.LdaMulticore.load('store/corpus.lda')
        topic_dist = {}
        for topic_num in range(lda.num_topics):
            topic_terms = lda.get_topic_terms(topic_num, topn=None)
            distribution = tuple([topic_term[1] for topic_term in topic_terms])
            topic_dist[topic_num] = distribution
        topic_dist_values = np.array(list(topic_dist.values()))
        with open(topic_dist_pickle, 'wb') as f:
            pickle.dump(topic_dist, f)
        with open(topic_dist_values_pickle, 'wb') as f:
            pickle.dump(topic_dist_values, f)
    return (topic_dist, topic_dist_values)

# topic agglomorative clustering
def cluster_topics(num_clusters, topic_dist_values):
    '''
    agglomerative clustering on set of inputs
    '''
    alg = AgglomerativeClustering(n_clusters=num_clusters)
    alg.fit(topic_dist_values)
    clusters = alg.fit_predict(topic_dist_values)
    return clusters

# unsupervised cluster validation measure
def cluster_validation(topic_dist_values, cluster_labels):
    '''
    calculates silhouette, calinski_harabaz, and dunn scores for cluster quality measurement
    '''
    scores = {}
    scores['silhouette'] = silhouette_score(topic_dist_values, cluster_labels)
    scores['ch'] = calinski_harabaz_score(topic_dist_values, cluster_labels)
    scores['dunn'] = dunn_fast(topic_dist_values, cluster_labels)
    return scores

# calculate dissimilarity matrix for MDS
def calc_distance(topics, n, shared_list, i):
    tmp = shared_list[i]
    for j in range(n):
        topic_i = topics[i]
        topic_j = topics[j]
        distance = Distance(topic_i, topic_j).kl()
        distance_r = Distance(topic_j, topic_i).kl()
        tmp[j] = (distance + distance_r) / 2.0
    shared_list[i] = tmp

def dissim(topics, replot=None, pickle_enabled=True, repickle=False):
    filename = 'dissim'
    if replot is not None:
        filename += replot
    if pickle_enabled and not repickle:
        obj = pickle_load(filename)
        if obj is not None:
            return obj

    n = len(topics)
    manager = mp.Manager()
    shared_list = manager.list([[0 for x in range(n)] for x in range(n)])

    p = mp.Pool(os.cpu_count())
    func = partial(calc_distance, topics, n, shared_list)
    p.map(func, range(n))
    p.close()
    p.join()

    if pickle_enabled:
        pickle_store(filename, list(shared_list))
    return list(shared_list)

# stress calculation
def calc_stress_helper(dist_matrix, eucl_matrix, n, shared_value, i):
    partial_sum = 0
    dist_row = dist_matrix[i]
    points_row = eucl_matrix[i]
    for j in range(n):
        if i != j:
            partial_sum += pow((dist_row[j] - points_row[j]), 2)
    shared_value.set(shared_value.get() + partial_sum)

def calc_stress(dist_matrix, eucl_matrix):
    '''
    my own stress calculation
    '''
    stress = ((eucl_matrix.ravel() - dist_matrix.ravel()) ** 2).sum() / 2
    return stress

def list_stress_points(dist_matrix, eucl_matrix, pickle_enabled=True):
    filename = 'stress_points'
    if pickle_enabled:
        obj = pickle_load(filename)
        if obj is not None:
            return obj

    N = len(dist_matrix)
    points_stress = []
    for i in range(N):
        for j in range(i):
            dist = dist_matrix[i][j]
            eucl = eucl_matrix[i][j]
            stress = math.fabs(dist - eucl)
            points_stress.append((i, j, stress))

    points_stress_sorted = sorted(points_stress, key=lambda tup: tup[1])
    if pickle_enabled:
        pickle_store(filename, points_stress_sorted)
    return points_stress_sorted

def calc_total_stress_points(dist_matrix, eucl_matrix, replot=None, point_indicies=None):
    '''
    Calculate total stress value for each point
    '''
    filename = 'total_stress_points'
    if replot is not None:
        filename += replot
    obj = pickle_load(filename)
    if obj is not None:
        return obj

    N = len(dist_matrix)
    total_stress_points = []
    for i, point in enumerate(eucl_matrix):
        dist = dist_matrix[i]
        total_stress = ((point - dist) ** 2).sum()
        index = i
        if point_indicies is not None:
            index = point_indicies[index]
        total_point_stress = (index, total_stress)
        total_stress_points.append(total_point_stress)

    total_stress_points_sorted = sorted(total_stress_points, key=lambda tup: tup[1])
    pickle_store(filename, total_stress_points_sorted)
    return total_stress_points_sorted

def get_quads(points_stress_list):
    '''
    segments an array of (point, total_stress) into four quadrants by std
    '''
    stress_list = np.array([point_total_stress[1] for point_total_stress in points_stress_list])
    mean = stress_list.mean()
    std = stress_list.std()

    first = mean - std
    second = mean
    third = mean + std

    quads = [[], [], [], []]
    for points_stress in points_stress_list:
        stress = points_stress[1]
        if stress < first:
            quads[0].append(points_stress)
        elif stress < second:
            quads[1].append(points_stress)
        elif stress < third:
            quads[2].append(points_stress)
        else:
            quads[3].append(points_stress)

    return quads

# stress matrix
def calc_stress_matrix(dist_matrix, eucl_matrix):
    filename = 'stress_matrix'
    obj = pickle_load(filename)
    if obj is not None:
        return obj

    N = len(dist_matrix)
    stress_matrix = []
    for i in range(N):
        stress_row = []
        for j in range(N):
            stress = math.pow((dist_matrix[i][j] - eucl_matrix[i][j]) ** 2, 0.5)
            stress_row.append(stress)
        stress_matrix.append(stress_row)

    pickle_store(filename, stress_matrix)
    return stress_matrix

# shepard plot calculation
def calc_shepard(dist_matrix, eucl_matrix):
    '''
    shepard plot
    '''
    obj = pickle_load('shepard')
    if obj is not None:
        return obj

    n = len(eucl_matrix)

    shepard_points = []
    for i in range(n):
        for j in range(i):
            if i != j:
                original_distance = dist_matrix[i][j]
                reduced_distance = eucl_matrix[i][j]
                shepard_points.append([original_distance, reduced_distance])

    obj = np.array(shepard_points)
    pickle_store('shepard', obj)
    return obj

# checksum calculation
def calc_checksum(dist_matrix):
    '''
    verify that the matrix are actually the same
    '''
    return sum([sum(dist_matrix[i]) for i in range(len(dist_matrix))])

# mds helper
def mds_helper(dist_matrix, r=2, replot=None, pickle_enabled=True):
    '''
    pickle_enabled: if False, compute everytime and don't pickle anything
    '''
    filename = 'mds%s' % r
    if replot is not None:
        filename += replot
    if pickle_enabled:
        obj = pickle_load(filename)
        if obj is not None:
            return obj

    mds = MDS(n_components=r, n_jobs=-1, dissimilarity='precomputed', random_state=tvconf.SEED, verbose=1)
    points = mds.fit_transform(dist_matrix)
    print('MDS Stress: %.10f' % mds.stress_)
    if pickle_enabled:
        pickle_store(filename, points)
    return points

def mds_r_calc(dist_matrix, final_r):
    '''
    calculate stress vs. mds_final_r to find optimal final_r (lower_r in order to visualize better with high dimensional input data)
    '''
    stress_r = []
    for r in range(1, final_r):
        mds = MDS(n_components=r, n_jobs=-1, dissimilarity='precomputed', random_state=tvconf.SEED, verbose=1)
        points = mds.fit_transform(dist_matrix)
        stress_r.append(mds.stress_)
    return stress_r

# euclidean distance

# graph helper methods
def find_k(stress_points):
    '''
    returns k cutoff value given every stress level between every pair of topics
    currently, it is (mean - std)
    stress_points: [(<topic_1>, <topic_2>, <stress_value>)]
    '''
    stress_values = np.array([stress_point[2] for stress_point in stress_points])
    mean = stress_values.mean()
    std = stress_values.std()
    k = mean + std
    return k

# iterative mds - DEPRECATE
def preprocess(quads):
    '''
    array of arrays => array
    '''
    all_quads = quads[0] + quads[1]
    points = [quad[0] for quad in all_quads]
    return points

def replot_topics(replot_points, topic_dist_values):
    replot_topic_dist_values = []
    for replot_point in replot_points:
        replot_topic_dist_values.append(topic_dist_values[replot_point])
    return np.array(replot_topic_dist_values)

def retrieve_good_points(quads, points, point_indicies=None):
    good_points = []
    for point_stress in quads:
        index = point_stress[0]
        if point_indicies is not None:
            index = point_indicies.index(index)
        eucl_point = points[index]
        good_points.append((point_stress[0], eucl_point))
    return good_points
