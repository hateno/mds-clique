import logging, os, json, tvconf, argparse, math
import numpy as np

from distance import Distance
from sim.fastmap import distmatrix, fastmap
from sim.graphit import plot
from sim.mycorpus import MyCorpus
from sim.topics import calc_shepard, calc_stress, cluster_topics, cluster_validation, dissim, load_topics
from sklearn.manifold import MDS

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
seed = np.random.RandomState(seed=5)

parser = argparse.ArgumentParser(description='Perform different functions')
parser.add_argument('type', type=str)
args = parser.parse_args()

(topic_dist, topic_dist_values) = load_topics()

if args.type == 'fastmap-kl':
    dist_matrix = distmatrix(list( range( len( topic_dist_values ) ) ), c = lambda x, y: Distance(topic_dist[x], topic_dist[y]).kl())
    points = fastmap(dist_matrix, 2)
elif args.type == 'fastmap-rkl':
    dist_matrix = distmatrix(list( range( len( topic_dist_values ) ) ), c = lambda x, y: 1.0 / Distance(topic_dist[x], topic_dist[y]).kl())
    points = fastmap(dist_matrix, 2)
elif args.type == 'fastmap-hellinger':
    dist_matrix = distmatrix(list( range( len( topic_dist_values ) ) ), c = lambda x, y: Distance(topic_dist[x], topic_dist[y]).hellinger())
    points = fastmap(dist_matrix, 2)
elif args.type == 'fastmap-tvd':
    dist_matrix = distmatrix(list(range(len(topic_dist_values))), c = lambda x, y: Distance(topic_dist[x], topic_dist[y]).tvd())
    points = fastmap(dist_matrix, 2)
elif args.type == 'mds':
    dist_matrix = dissim(topic_dist_values)
    dist_check = sum([sum(dist_matrix[i]) for i in range(len(dist_matrix))])
    print('dist: ', str(dist_check))
    mds = MDS(n_jobs=-1, dissimilarity='precomputed', random_state=seed)
    points = mds.fit_transform(dist_matrix)
    print('tpoint: ', str(points[0]))

# calculate stress and shepard
stress = calc_stress(dist_matrix, points)
shepard = calc_shepard(dist_matrix, points)

# cluster and cluster scores
clusters = cluster_topics(10, topic_dist_values)
scores = cluster_validation(topic_dist_values, clusters)

# plot it
plot(points, clusters, point_clusters=False)
