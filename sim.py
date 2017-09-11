import logging, os, json, tvconf, argparse, math, sys
import numpy as np
import sim.dim, sim.output, sim.topics

from distance import Distance
from sim.fastmap import distmatrix, fastmap
from sim.graphit import plot, plot_quads
from sim.mycorpus import MyCorpus
from sim.topics import calc_checksum, calc_shepard, calc_stress, calc_stress_matrix, calc_total_stress_points, cluster_topics, cluster_validation, dissim, get_quads, list_stress_points, load_topics, mds_helper
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
seed = np.random.RandomState(seed=5) # TODO port to ini file

parser = argparse.ArgumentParser(description='Perform different functions')
parser.add_argument('type', type=str)
parser.add_argument('--flush', action='store_true')
args = parser.parse_args()

(topic_dist, topic_dist_values) = load_topics()

if args.flush:
    sim.topics.flush()

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
    dist_matrix_list = dissim(topic_dist_values)
    dist_matrix = check_array(dist_matrix_list)
    #dist_check = calc_checksum(dist_matrix)
    #print('dist: ', str(dist_check))
    points = mds_helper(dist_matrix)
elif args.type == 'mds3':
    dist_matrix_list = dissim(topic_dist_values)
    dist_matrix = check_array(dist_matrix_list)
    #dist_check = calc_checksum(dist_matrix)
    #print('dist: ', str(dist_check))
    points = mds_helper(dist_matrix, r=3)

# mds-r vs stress calculation
'''
stress_r = []
for r in range(1, 80):
    mds = MDS(n_components=r, n_jobs=-1, dissimilarity='precomputed', random_state=seed, verbose=1)
    points = mds.fit_transform(dist_matrix)
    stress_r.append(mds.stress_)
'''

# TODO add switch logic to section below

# calculate stress and shepard
eucl_matrix = euclidean_distances(points)
stress = calc_stress(dist_matrix, eucl_matrix)
shepard = calc_shepard(dist_matrix, eucl_matrix)
stress_points = list_stress_points(dist_matrix, eucl_matrix)
total_stress_points = calc_total_stress_points(dist_matrix, eucl_matrix)

# cluster and cluster scores
clusters = cluster_topics(10, topic_dist_values)
scores_topics = cluster_validation(points, clusters)
clusters_points = cluster_topics(10, points)
scores_points = cluster_validation(points, clusters_points)

# plot all points
#plot(points, clusters, point_clusters=False, three_d=True, title='MDS (r=3)', xlabel='x1', ylabel='x2', zlabel='x3')

# plot each quadrant of points
#quads_list = plot_quads(points, clusters, stress_points)
# compute mds on each quadrant
#sim.dim.calc_mds_quads(quads_list, topic_dist_values, clusters)

# output to stress.json
stress_matrix = calc_stress_matrix(dist_matrix, eucl_matrix)
sim.output.output_json(stress_matrix, points, clusters)

# output to stress.html
#sim.output.output_html(stress_matrix, points, clusters)
quads = get_quads(total_stress_points)
sim.output.output_html_list(stress_points, total_stress_points, quads)

# iterative mds - DEPRECATE
final_mds = []

i = 0
current_points = points
replot_points = None
while quads[3] != []:
    good_points = sim.topics.retrieve_good_points(quads[0] + quads[1], current_points, point_indicies=replot_points)
    if good_points == []:
        final_mds.append(sim.topics.retrieve_good_points(quads[2] + quads[3], current_points, point_indicies=replot_points))
        break
    final_mds.append(good_points)

    replot_points = sim.topics.preprocess(quads[2:])
    replot_topic_dist_values = sim.topics.replot_topics(replot_points, topic_dist_values)
    replot_dist_matrix_list = dissim(replot_topic_dist_values, replot='-%s' % i)
    replot_dist_matrix = check_array(replot_dist_matrix_list)
    current_points = mds_helper(replot_dist_matrix, replot='-%s' % i)

    replot_eucl_matrix = euclidean_distances(current_points)
    replot_total_stress_points = calc_total_stress_points(replot_dist_matrix, replot_eucl_matrix, replot='-%s' % i, point_indicies=replot_points)
    quads = get_quads(replot_total_stress_points)
    i += 1
