import networkx as nx
import numpy as np
import sim.topics

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

class MGraph:
    MAX_ITER = 20 # hasn't been hit so far, just a safeguard

    def __init__(self, points, stress_points, topic_dist_values, dist_matrix, eucl_matrix, r_dim, clusters):
        self.clusters = clusters
        self.points = points
        self.stress_points = stress_points
        self.graphs = []
        self.topic_dist_values = topic_dist_values
        self.dist_matrix = dist_matrix
        self.eucl_matrix = eucl_matrix
        self.r_dim = r_dim

        # generate graph M
        self.M = nx.Graph()
        for stress_point in self.stress_points:
            source = stress_point[0]
            target = stress_point[1]
            if source not in self.M:
                point_x = points[source][0]
                point_y = points[source][1]
                self.M.add_node(source, x=point_x, y=point_y)
            if target not in self.M:
                point_x = points[target][0]
                point_y = points[target][1]
                self.M.add_node(target, x=point_x, y=point_y)
            mds_distance = eucl_matrix[source][target]
            actual_distance = dist_matrix[source][target]
            self.M.add_edge(source, target, weight=stress_point[2], mds_distance=mds_distance, actual_distance=actual_distance)
        self.graphs.append(self.M)

    def _comparator_distance(self, edge_data, k):
        mds_distance = edge_data['mds_distance']
        actual_distance = edge_data['actual_distance']
        return abs(mds_distance - actual_distance) / actual_distance > k

    def _comparator_stress(self, edge_data, k):
        weight = edge_data['weight']
        return weight > k

    def find_cliques_distance(self, k):
        return self._find_cliques(self._comparator_distance, pk=k)

    def find_cliques_stress(self):
        return self._find_cliques(self._comparator_stress)

    def _find_cliques(self, comparator, pk=None):
        '''
        pk, used only for find_cliques_stress
        comparator, criteria based on which edge to remove based on edge data
        '''
        M = self.M
        points = self.points
        cliques = []
        if pk == None:
            k = sim.topics.find_k(self.stress_points)
        else:
            k = pk

        # mds-graph algorithm
        iteration = 0
        indicies = None
        while len(M) != 0 and iteration < self.MAX_ITER:
            print('mds-graph iteration: %s' % iteration)
            # generate graph M1, remove edges within distance k factor
            Mp = M.copy()
            for edge in Mp.edges():
                edge_data = Mp.get_edge_data(edge[0], edge[1])
                if comparator(edge_data, k):
                    Mp.remove_edge(edge[0], edge[1])

            cliques_list = list(nx.find_cliques(Mp))
            max_clique_size = max([len(clique) for clique in cliques_list])
            maximal_cliques = [clique for clique in cliques_list if len(clique) == max_clique_size] # is there only one maximal clique? why so much overlap?

            # store these vertices and their points
            clique_vertices = set(np.concatenate(maximal_cliques))
            clique_points = []
            for vertex in clique_vertices:
                vertex_index = vertex
                if indicies != None:
                    vertex_index = indicies.index(vertex)
                point = points[vertex_index]
                clique_points.append((vertex, point))
            cliques.append(clique_points)

            # consider all vertices in M but not in C (V1 = M - C)
            Mn = M.copy()
            Mn.remove_nodes_from(clique_vertices)
            if len(Mn) == 0:
                print('Empty graph! Exiting...')
                break

            # extract topics that correspond to V1
            new_topics_indicies = [i for i in range(len(self.topic_dist_values)) if i in Mn.nodes()]
            new_topics_values = np.array([self.topic_dist_values[new_topics_index] for new_topics_index in new_topics_indicies])

            # run MDS on new_topic_values
            new_dist_matrix_list = sim.topics.dissim(new_topics_values, pickle_enabled=False)
            new_dist_matrix = check_array(new_dist_matrix_list)
            new_points = sim.topics.mds_helper(new_dist_matrix, r=self.r_dim, pickle_enabled=False)
            new_eucl_matrix = euclidean_distances(new_points)
            new_stress_points = sim.topics.list_stress_points(new_dist_matrix, new_eucl_matrix, pickle_enabled=False)

            # generate graph Mi
            Mi = nx.Graph()
            for i, new_stress_point in enumerate(new_stress_points):
                source_index = new_stress_point[0]
                source = new_topics_indicies[source_index]
                target_index = new_stress_point[1]
                target = new_topics_indicies[target_index]
                if source not in Mi:
                    point_x = new_points[source_index][0]
                    point_y = new_points[source_index][1]
                    Mi.add_node(source, x=point_x, y=point_y)
                if target not in Mi:
                    point_x = new_points[target_index][0]
                    point_y = new_points[target_index][1]
                    Mi.add_node(target, x=point_x, y=point_y)
                mds_distance = new_eucl_matrix[source_index][target_index]
                actual_distance = new_dist_matrix[source_index][target_index]
                Mi.add_edge(source, target, weight=new_stress_point[2], mds_distance=mds_distance, actual_distance=actual_distance)
            self.graphs.append(Mi)

            # set values for next iteration
            M = Mi
            points = new_points
            indicies = new_topics_indicies
            if pk == None:
                k = sim.topics.find_k(new_stress_points)

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
                for clique_point in clique_points:
                    topic = clique_point[0]
                    cluster = self.clusters[topic]
                    point = clique_point[1]
                    f.write('%s %s %s\n' % (topic, cluster, ' '.join(str(coord) for coord in point)))

        with open('out/data', 'w') as f:
            for topic, point in enumerate(self.topic_dist_values):
                cluster = self.clusters[topic]
                f.write('%s %s %s\n' % (topic, cluster, ' '.join(str(coord) for coord in point)))
