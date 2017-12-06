import ctypes as c
import numpy as np
import multiprocessing as mp
import logging, math, os, pickle, shutil, tvconf
import sim.relative

from functools import partial
from gensim import models
from sim.distance import Distance
from sim.jqmcvi import dunn_fast
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from multiprocessing.pool import ThreadPool

logger = logging.getLogger()

# marshal utility functions
def flush():
    '''
    Clear all pickle cache
    '''
    print('deleting tmp/flush...')
    shutil.rmtree('tmp/flush', ignore_errors=True)

def pickle_load(ident, filename):
    path = 'store/ident/%s' % ident
    if not os.path.exists(path):
        os.makedirs(path)
    obj = None
    filepath = '%s/%s' % (path, filename)
    if os.path.exists(filepath):
        logger.info('unpickling %s, %s...' % (ident, filename))
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
    else:
        logger.info('pickle %s, %s not found, calculating...' % (ident, filename))
    return obj

def pickle_store(ident, filename, obj):
    filepath = 'store/ident/%s/%s' % (ident, filename)
    with open(filepath, 'wb') as f:
        logger.info('pickling %s, %s...' % (ident, filename))
        pickle.dump(obj, f)

def load_topics():
    '''
    load from ap store
    '''
    print('Loading from store...')
    lda = models.LdaMulticore.load('store/corpus.lda')
    topic_dist = {}
    for topic_num in range(lda.num_topics):
        topic_terms = lda.get_topic_terms(topic_num, topn=None)
        distribution = tuple([topic_term[1] for topic_term in topic_terms])
        topic_dist[topic_num] = distribution
    topic_dist_values = np.array(list(topic_dist.values()))
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

def _shrink_matrix(matrix, index):
    d = np.take(matrix, index, axis=0)
    dd = np.take(d, index, axis=1)
    return dd

def dissim(topics, replot=None, ident=None, index=None):
    if topics[0][0] is None: # TODO hack solution, simply a flag for dissim to fetch precomputed dissim and perform _shrink_matrix operation if necessary
        dissim = topics[0][1] # fetch precomputed matrix (stored in every array position)
        if index is not None:
            dissim = _shrink_matrix(dissim, index) # if index argument exists, shrink precomputed matrix
        return dissim
    filename = 'dissim'
    if ident is not None and ident != '':
        obj = pickle_load(ident, filename)
        if obj is not None:
            return obj

    n = len(topics)
    shared_list = np.zeros([n, n])

    p = ThreadPool(os.cpu_count() // 2)
    func = partial(calc_distance, topics, n, shared_list)
    p.map(func, range(n))
    p.close()
    p.join()

    if ident is not None and ident != '':
        pickle_store(ident, filename, list(shared_list))
    return shared_list

def dissim_single(topics, index=None):
    '''
    calculates dissimilarity matrix same as dissim(), but only a single job
    '''
    if topics[0][0] is None: # dist matrix precomputed
        dissim = topics[0][1]
        if index is not None:
            dissim = _shrink_matrix(dissim, index)
        return dissim
    n = len(topics)
    dissim = np.zeros([n, n])
    for i in range(n):
        for j in range(i):
            topic_i = topics[i]
            topic_j = topics[j]
            distance = Distance(topic_i, topic_j).kl()
            distance_r = Distance(topic_j, topic_i).kl()
            dissim[i][j] = (distance + distance_r) / 2.0
            dissim[j][i] = (distance + distance_r) / 2.0
    return dissim

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

def calc_stress_matrix(dist_matrix, eucl_matrix):
    stress_matrix = np.absolute(np.divide((dist_matrix - eucl_matrix), eucl_matrix))
    return stress_matrix

def _list_stress_points(i, mpoints_stress, dist_matrix, eucl_matrix):
    for j in range(i):
        dist = dist_matrix[i][j]
        eucl = eucl_matrix[i][j]
        stress = math.fabs(dist - eucl)
        mpoints_stress.append((i, j, stress))

def list_stress_points(dist_matrix, eucl_matrix, ident=None, parallel=False):
    filename = 'stress_points'
    if ident is not None:
        obj = pickle_load(ident, filename)
        if obj is not None:
            return obj

    N = len(dist_matrix)
    points_stress = []
    if not parallel:
        for i in range(N):
            for j in range(i):
                dist = dist_matrix[i][j]
                eucl = eucl_matrix[i][j]
                stress = math.fabs((dist - eucl) / eucl)
                points_stress.append((i, j, stress))
    else:
        N_PROCESS = os.cpu_count() // 2
        mpoints_stress = []
        pool = ThreadPool(N_PROCESS)
        pargs = [(i, mpoints_stress, dist_matrix, eucl_matrix) for i in range(N)]
        pool.starmap(_list_stress_points, pargs)
        pool.close()
        pool.join()
        points_stress = list(mpoints_stress)

    points_stress_sorted = sorted(points_stress, key=lambda tup: tup[1])
    if ident is not None:
        pickle_store(ident, filename, points_stress_sorted)
    return points_stress_sorted

def calc_total_stress_points(dist_matrix, eucl_matrix, point_indicies=None, ident=None):
    '''
    Calculate total stress value for each point
    '''
    filename = 'total_stress_points'
    if ident is not None:
        obj = pickle_load(ident, filename)
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
    if ident is not None:
        pickle_store(ident, filename, total_stress_points_sorted)
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

# shepard plot calculation
def calc_shepard(dist_matrix, eucl_matrix, ident=None):
    '''
    shepard plot
    '''
    if ident is not None:
        obj = pickle_load(ident, 'shepard')
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
    if ident is not None:
        pickle_store(ident, 'shepard', obj)
    return obj

# checksum calculation
def calc_checksum(dist_matrix):
    '''
    verify that the matrix are actually the same
    '''
    return sum([sum(dist_matrix[i]) for i in range(len(dist_matrix))])

# mds helper
def mds_helper(dist_matrix, rdim=2, verbose=0, n_jobs=-1, ident=None):
    filename = 'mds%s' % rdim
    if ident is not None:
        obj = pickle_load(ident, filename)
        if obj is not None:
            return obj

    mds = MDS(n_components=rdim, n_jobs=n_jobs, dissimilarity='precomputed', random_state=tvconf.SEED, verbose=verbose)
    points = mds.fit_transform(dist_matrix)
    stress = sim.relative.Relative.calc_stress(points, dist_matrix)
    logging.info('mds done, stress: %.4f' % stress)

    if ident is not None:
        pickle_store(ident, filename, points)

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
    k = mean
    if k < 0:
        return 0
    return k

def calc_aspe(stress, n):
    n_edges = n * (n - 1) / 2
    if n_edges == 0:
        return 0
    return stress / math.sqrt(n_edges)

def calc_aspe_cliques(cliques_stress, cliques_points):
    result = []
    for clique_stress, clique_points in zip(cliques_stress, cliques_points):
        n = len(clique_points)
        aspe = 0
        if n > 1:
            aspe = clique_stress / math.sqrt(n * (n - 1) / 2)
        result.append(aspe)
    return result

# output mds points output to file
def print_mds_points(points, clusters=None, filename=None, subdir=None, index=None):
    _filename = 'points' if filename is None else filename
    _subdir = 'out/experiment' if subdir is None else 'out/experiment/%s' % subdir
    if not os.path.exists(_subdir):
        os.mkdir(_subdir)
    path = '%s/%s' % (_subdir, _filename)
    logger.info('printing mds points to %s...' % path)
    with open(path, 'w') as f:
        for topic, point in enumerate(points):
            idx = topic
            if index is not None:
                idx = index[topic]
            if clusters is not None:
                cluster = clusters[topic]
                f.write('%s %s %s\n' % (idx, cluster, ' '.join(str(coord) for coord in point)))
            else:
                f.write('%s %s\n' % (idx, ' '.join(str(coord) for coord in point)))

# numpy save/load helper
def np_save(filename, var):
    with open('out/numpy/%s' % filename, 'wb') as f:
        np.save(f, var)

def np_load(filename):
    var = None
    with open('out/numpy/%s' % filename, 'rb') as f:
        var = np.load(f)
    return var

def get_topic_dist_cluster(n_data, n_clusters, ident=None):
    '''
    if ident is None, then generate a new one, else retrieve if it's previously stored or save it as ident
    '''
    if ident is not None:
        path = 'store/ident/%s' % ident
        if not os.path.exists(path):
            os.mkdir(path)
            with open('%s/topic_dist_values', 'wb') as f:
                np.save(f, topic_dist_values)
            with open('%s/dist_matrix', 'wb') as f:
                np.save(f, dist_matrix)
            with open('%s/clusters', 'wb') as f:
                np.save(f, clusters)

    else:
        (topic_dist_values, dist_matrix, clusters) = sim.util.generate_data(n=n_data, dim=3, n_clusters=n_clusters)

# print 3d topic dist values
def print_topic_dist_values(topic_dist_values, clusters, subdir):
    subdir = 'out/experiment/%s' % subdir
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    path = '%s/data' % subdir
    with open(path, 'w') as f:
        logger.info('printing topic_dist_values to %s...' % path)
        for i, topic_dist_value in enumerate(topic_dist_values):
            f.write('%s %s %s %s %s\n' % (i, clusters[i], topic_dist_value[0], topic_dist_value[1], topic_dist_value[2]))
