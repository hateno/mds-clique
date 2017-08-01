import logging, os, json, tvconf, argparse, math
import lib.fastmap

from distance import Distance
from sim.util import dissim
from sim.mycorpus import MyCorpus
from sim.topics import load_topics, cluster_topics
from sklearn.manifold import MDS

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Perform different functions')
parser.add_argument('type', type=str)
args = parser.parse_args()

topic_dist_values = load_topics()
clusters = cluster_topics(10, topic_dist_values)

if args.type == 'fastmap-kl':
    dist = lib.fastmap.distmatrix( list( range( len( topic_dist_values ) ) ), c = lambda x, y: Distance(topic_dist[x], topic_dist[y]).kl() )
    p = lib.fastmap.fastmap( dist, 2 )
    print(p)
elif args.type == 'fastmap-rkl':
    dist = lib.fastmap.distmatrix( list( range( len( topic_dist_values ) ) ), c = lambda x, y: 1.0 / Distance(topic_dist[x], topic_dist[y]).kl() )
    p = lib.fastmap.fastmap( dist, 2 )
    print(p)
elif args.type == 'fastmap-hellinger':
    dist = lib.fastmap.distmatrix( list( range( len( topic_dist_values ) ) ), c = lambda x, y: Distance(topic_dist[x], topic_dist[y]).hellinger() )
    p = lib.fastmap.fastmap( dist, 2 )
    print(p)
elif args.type == 'mds':
    dist = dissim( topic_dist_values )
    mds = MDS( n_jobs=-1, dissimilarity='precomputed' )
    p = mds.fit_transform( dist )
