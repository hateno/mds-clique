import numpy as np

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
