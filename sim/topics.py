import ctypes as c
import numpy as np
import multiprocessing as mp
import os, pickle

from distance import Distance
from functools import partial
from gensim import models
from sim.jqmcvi import dunn_fast
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabaz_score

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
    alg = AgglomerativeClustering(n_clusters=num_clusters)
    alg.fit(topic_dist_values)
    clusters = alg.fit_predict(topic_dist_values)
    return clusters

# unsupervised cluster validation measure
def cluster_validation(topic_dist_values, cluster_labels):
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
        distance = Distance( topic_i, topic_j )
        tmp[j] = distance.tvd()
    shared_list[i] = tmp

def dissim(topics):
    n = len(topics)
    manager = mp.Manager()
    shared_list = manager.list([[0 for x in range(n)] for x in range(n)])

    p = mp.Pool(os.cpu_count())
    func = partial(calc_distance, topics, n, shared_list)
    p.map(func, range(n))
    p.close()
    p.join()

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
    stress = ((eucl_matrix.ravel() - dist_matrix.ravel()) ** 2).sum() / 2
    return stress

def list_stress_points(dist_matrix, eucl_matrix):
    points_stress = []
    for i, point in enumerate(eucl_matrix):
        dist = dist_matrix[i]
        stress = ((point - dist) ** 2).sum() / 2
        point_stress = (i, stress)
        points_stress.append(point_stress)

    points_stress_sorted = sorted(points_stress, key=lambda tup: tup[1])
    return points_stress_sorted

# shepard plot calculation
def calc_shepard(dist_matrix, eucl_matrix):
    n = len(eucl_matrix)

    shepard_points = []
    for i in range(n):
        for j in range(i):
            if i != j:
                original_distance = dist_matrix[i][j]
                reduced_distance = eucl_matrix[i][j]
                shepard_points.append([original_distance, reduced_distance])

    return np.array(shepard_points)
