import ctypes as c
import numpy as np
import multiprocessing as mp

from distance import Distance
from functools import partial
from gensim import models
from sklearn.cluster import AgglomerativeClustering

def load_topics():
    lda = models.LdaMulticore.load('store/corpus.lda')
    topic_dist = {}
    for topic_num in range(lda.num_topics):
        topic_terms = lda.get_topic_terms(topic_num, topn=None)
        distribution = tuple([topic_term[1] for topic_term in topic_terms])
        topic_dist[topic_num] = distribution
    topic_dist_values = np.array(list(topic_dist.values()))
    return topic_dist_values

def cluster_topics(num_clusters, topic_dist_values):
    alg = AgglomerativeClustering(n_clusters=num_clusters)
    alg.fit(topic_dist_values)
    clusters = alg.fit_predict(topic_dist_values)
    return clusters

def calc_distance(topics, n, shared_list, i):
    print(i)
    tmp = shared_list[i]
    for j in range(n):
        topic_i = topics[i]
        topic_j = topics[j]
        distance = Distance( topic_i, topic_j )
        tmp[j] = distance.tvd()
    shared_list[i] = tmp

# calculate dissimilarity matrix for MDS
def dissim(topics):
    n = len(topics)
    manager = mp.Manager()
    shared_list = manager.list([[0 for x in range(n)] for x in range(n)])

    p = mp.Pool(32)
    func = partial(calc_distance, topics, n, shared_list)
    p.map(func, range(n))
    p.close()
    p.join()

    return list(shared_list)
