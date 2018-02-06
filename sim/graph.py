import logging, os
import multiprocessing as mp
import networkx as nx
import numpy as np
import sim.topics, sim.relative

from multiprocessing import Manager, Pool
from multiprocessing.pool import ThreadPool
from networkx.algorithms.approximation.clique import clique_removal
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

logger = logging.getLogger()

class MGraph:
    def __init__(self, points, topic_dist_values, dist_matrix, eucl_matrix, rdim, clusters, ignore_points=None, approximation=False, parallel=False):
        self.clusters = clusters
        self.points = points
        self.topic_dist_values = topic_dist_values
        self.dist_matrix = dist_matrix
        self.eucl_matrix = eucl_matrix
        stress_matrix = sim.topics.calc_stress_matrix(dist_matrix, eucl_matrix)
        self.rdim = rdim
        self.ignore_points = ignore_points
        self.approximation = approximation
        self.edge_data = dict()
        self.parallel = parallel
        self.N_PROCESS = mp.cpu_count() // 8

        # generate graph M
        self.M = nx.Graph()
        ebunch = []
        edge_data = dict()
        if self.parallel:
            pool = ThreadPool(self.N_PROCESS)
            pargs = [(source, points, eucl_matrix, dist_matrix, stress_matrix, ignore_points,  edge_data, self.M) for source in range(len(dist_matrix))]
            pool.starmap(self.process_target, pargs)
            pool.close()
            pool.join()
        else:
            for source in range(len(dist_matrix)):
                self.process_target(source, points, eucl_matrix, dist_matrix, stress_matrix, ignore_points, edge_data, self.M)
        self.edge_data = dict(edge_data)

    def _comparator_distance(self, edge_data, k):
        if 'mds_distance' in edge_data and 'actual_distance' in edge_data:
            mds_distance = edge_data['mds_distance']
            actual_distance = edge_data['actual_distance']
            return abs(mds_distance - actual_distance) / mds_distance > k
        return False # don't remove this edge

    def _comparator_stress(self, edge_data, k):
        if 'weight' in edge_data:
            weight = edge_data['weight']
            return weight > k
        return False

    def find_cliques_distance(self, k, stop=None, combine=False):
        return self._find_cliques(self._comparator_distance, pk=k, stop=stop, combine=combine)

    def find_cliques_stress(self, combine=False):
        return self._find_cliques(self._comparator_stress, combine=combine)

    def _find_cliques(self, comparator, pk=None, stop=None, combine=False):
        '''
        pk, used only for find_cliques_stress
        comparator, criteria based on which edge to remove based on edge data
        '''
        M = self.M
        points = self.points
        cliques = []
        if pk == None:
            stress_points = sim.topics.list_stress_points(self.dist_matrix, self.eucl_matrix)
            k = sim.topics.find_k(stress_points)
        else:
            k = pk

        store = []

        # mds-graph algorithm
        iteration = 0
        indicies = None
        store = []

        while len(M) != 0:
            logger.info('mds-graph iteration: %s' % iteration)
            # generate graph M1, remove edges within distance k factor
            Mp = self.parallel_remove_edges(M, comparator, k)

            if self.approximation:
                logger.info('running approximation max clique...')
                #clique_vertices = max_clique(Mp)
                clique_vertices = clique_removal(Mp)
                clique_vertices = clique_vertices[1][0] # first element should be the maximum clique
                logger.info('clique found with length %d...' % len(clique_vertices))
            else:
                logger.info('running non-approx clique algorithm...')
                cliques_list = list(nx.find_cliques(Mp))
                max_clique_size = max([len(clique) for clique in cliques_list])
                maximal_cliques = [clique for clique in cliques_list if len(clique) == max_clique_size]
                clique_vertices = {}
                if combine:
                    logger.info('running clique algorithm in combination mode...')
                    clique_vertices = set(np.concatenate(maximal_cliques))
                elif len(maximal_cliques) > 0:
                    clique_vertices = set(maximal_cliques[0])

            # if one vertex is left out, go ahead and include that
            all_set = set(Mp.nodes())
            leftover_set = all_set - clique_vertices
            if len(leftover_set) == 1:
                clique_vertices |= leftover_set
                logger.info('%d is last vertex out, including that into clique...' % leftover_set.pop())

            # store these vertices and their points
            clique_points = []
            for vertex in clique_vertices:
                vertex_index = vertex
                if indicies != None:
                    vertex_index = indicies.index(vertex)
                point = points[vertex_index]
                clique_points.append((vertex, point))
            cliques.append(clique_points)
            if stop is not None and iteration >= stop:
                logger.info('stop limit reached, breaking out of graph-mds early')
                break

            # consider all vertices in M but not in C (V1 = M - C)
            Mn = M.copy()
            Mn.remove_nodes_from(clique_vertices)
            if len(Mn) == 0:
                logger.info('Empty graph! Exiting...')
                break

            # extract topics that correspond to V1
            new_topics_indicies = [i for i in range(len(self.topic_dist_values)) if i in Mn.nodes()]
            new_topics_values = np.array([self.topic_dist_values[new_topics_index] for new_topics_index in new_topics_indicies])

            # run MDS on new_topic_values
            new_dist_matrix_list = sim.topics.dissim(new_topics_values, new_topics_indicies, index=new_topics_indicies)
            new_dist_matrix = check_array(new_dist_matrix_list)
            new_points = sim.topics.mds_helper(new_dist_matrix, rdim=self.rdim)
            new_eucl_matrix = euclidean_distances(new_points)
            new_stress_matrix = sim.topics.calc_stress_matrix(new_dist_matrix, new_eucl_matrix)

            # generate graph Mi
            edge_data = dict()
            Mi = nx.Graph()
            if self.parallel:
                pool = ThreadPool(self.N_PROCESS)
                pargs = [(source_index, new_topics_indicies, new_points, new_eucl_matrix, new_dist_matrix, new_stress_matrix, edge_data, Mi) for source_index in range(len(new_dist_matrix))]
                pool.starmap(self.process_target_index, pargs)
                pool.close()
                pool.join()
            else:
                for source_index in range(len(new_dist_matrix)):
                    self.process_target_index(source_index, new_topics_indicies, new_points, new_eucl_matrix, new_dist_matrix, new_stress_matrix, edge_data, Mi)
            self.edge_data = dict(edge_data)

            # set values for next iteration
            M = Mi
            points = new_points
            indicies = new_topics_indicies
            if pk == None:
                new_stress_points = sim.topics.list_stress_points(new_dist_matrix, new_eucl_matrix, parallel=self.parallel)
                k = sim.topics.find_k(new_stress_points)
                #logger.warn('NEW K-VALUE: %.8f' % k);

            iteration += 1

        self.cliques = cliques
        return cliques

    def find_cluster_scores(self):
        '''
        return cluster scores for each clique
        '''
        cliques = self.cliques
        scores = []
        for clique in cliques:
            points = []
            points_cluster = []
            for point in clique:
                points.append(point[1])
                points_cluster.append(self.clusters[point[0]])

            try:
                clique_score = sim.topics.cluster_validation(points, points_cluster)
                scores.append(clique_score)
            except ValueError: # too few clusters
                scores.append(None)

        return scores

    def write_to_file(self):
        '''
        output cliques to file out_cliques_<X>
        '''
        for i, clique_points in enumerate(self.cliques):
            with open('out/cliques_%s' % i, 'w') as f:
                logger.info('writing clique %d to out/cliques_%d...' % (i, i))
                for clique_point in clique_points:
                    topic = clique_point[0]
                    cluster = self.clusters[topic]
                    point = clique_point[1]
                    f.write('%s %s %s\n' % (topic, cluster, ' '.join(str(coord) for coord in point)))

        logger.info('writing topic_dist_values to file out/data...')
        with open('out/data', 'w') as f:
            for topic, point in enumerate(self.topic_dist_values):
                cluster = self.clusters[topic]
                f.write('%s %s %s\n' % (topic, cluster, ' '.join(str(coord) for coord in point)))

    def parallel_remove_edges(self, M, comparator, k):
        Mp = M.copy()
        old_num_edges = len(Mp.edges())
        if self.parallel:
            N_PROCESS = mp.cpu_count() // 2
            pool = ThreadPool(self.N_PROCESS)
            pargs = [(edge, comparator, k, Mp) for edge in M.edges()]
            pool.starmap(self.parallel_remove_edges_helper, pargs)
            pool.close()
            pool.join()
        else:
            for edge in M.edges():
                self.parallel_remove_edges_helper(edge, comparator, k, Mp)
        new_num_edges = len(Mp.edges())
        diff_num_edges = old_num_edges - new_num_edges
        logger.info('removed %d edges (%.4f)...' % (diff_num_edges, diff_num_edges / old_num_edges))
        return Mp

    def parallel_remove_edges_helper(self, edge, comparator, k, Mp):
        if edge[0] == edge[1]:
            return
        edge_data = self.edge_data[(edge[0], edge[1])]
        if comparator(edge_data, k):
            Mp.remove_edge(edge[0], edge[1])

    def process_target_index(self, source_index, new_topics_indicies, new_points, new_eucl_matrix, new_dist_matrix, new_stress_matrix, edge_data, Mi):
        for target_index in range(source_index):
            source = new_topics_indicies[source_index]
            target = new_topics_indicies[target_index]
            mds_distance = new_eucl_matrix[source_index][target_index]
            actual_distance = new_dist_matrix[source_index][target_index]
            stress = new_stress_matrix[source_index][target_index]
            #ebunch.append((source, target))
            Mi.add_edge(source, target)
            edge_data[(source, target)] = {'weight': stress, 'mds_distance': mds_distance, 'actual_distance': actual_distance}
            edge_data[(target, source)] = {'weight': stress, 'mds_distance': mds_distance, 'actual_distance': actual_distance}

    def process_target(self, source, points, eucl_matrix, dist_matrix, stress_matrix, ignore_points, edge_data, M):
        for target in range(source):
            if ignore_points is not None and source in ignore_points and target in ignore_points:
                M.add_edge(source, target)
            else:
                mds_distance = eucl_matrix[source][target]
                actual_distance = dist_matrix[source][target]
                stress = stress_matrix[source][target]
                M.add_edge(source, target)
                edge_data[(source, target)] = {'weight': stress, 'mds_distance': mds_distance, 'actual_distance': actual_distance}
                edge_data[(target, source)] = {'weight': stress, 'mds_distance': mds_distance, 'actual_distance': actual_distance}
