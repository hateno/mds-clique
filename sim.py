import logging, os, json, tvconf, argparse, math, sys
import networkx as nx
import numpy as np
import sim.dim, sim.graph, sim.output, sim.topics

from sim.graphit import plot, plot_quads
from sim.mycorpus import MyCorpus
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Perform different functions')
parser.add_argument('type', type=str)
parser.add_argument('--flush', action='store_true')
args = parser.parse_args()

(topic_dist, topic_dist_values) = sim.topics.load_topics()

# clear out all pickle storage
if args.flush:
    sim.topics.flush()

# decide MDS type
r_dim = None
if args.type == 'mds':
    r_dim = 2
elif args.type == 'mds3':
    r_dim = 3

dist_matrix_list = sim.topics.dissim(topic_dist_values, repickle=True)
dist_matrix = check_array(dist_matrix_list)
points = sim.topics.mds_helper(dist_matrix, r=r_dim)

eucl_matrix = euclidean_distances(points)

# calculate stress between every point pairs
stress_points = sim.topics.list_stress_points(dist_matrix, eucl_matrix)

# clusters of mds
clusters = sim.topics.cluster_topics(10, topic_dist_values)

# mds graph
mgraph = sim.graph.MGraph(points, stress_points, topic_dist_values, r_dim)
cliques = mgraph.find_cliques()

# output cliques to file
for i, clique_points in enumerate(cliques):
    with open('out_cliques_%s' % i, 'w') as f:
        for clique_point in clique_points:
            f.write('%s %s %s\n' % (clique_point[0], clique_point[1], clique_point[2]))
