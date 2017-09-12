import networkx as nx
import numpy as np
import sim.topics

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

class MGraph:
    MAX_ITER = 20

    def __init__(self, points, stress_points, topic_dist_values, r_dim):
        self.points = points
        self.stress_points = stress_points
        self.graphs = []
        self.topic_dist_values = topic_dist_values
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
            self.M.add_edge(source, target, weight=stress_point[2])
        self.graphs.append(self.M)

    def find_cliques(self):
        M = self.M
        points = self.points

        k = sim.topics.find_k(self.stress_points)

        cliques = []

        # mds-graph algorithm
        iterations = 0
        indicies = None
        while len(M) != 0 or iterations < self.MAX_ITER:
            print('mds-graph iteration: %s' % iterations)
            # generate graph M1, remove edges with stress > k
            Mp = M.copy()
            for edge in Mp.edges():
                edge_data = Mp.get_edge_data(edge[0], edge[1])
                weight = edge_data['weight']
                if weight > k:
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
                point_x = point[0]
                point_y = point[1]
                clique_points.append((vertex, point_x, point_y))
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
            new_k = sim.topics.find_k(new_stress_points)

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
                Mi.add_edge(source, target, weight=new_stress_point[2])
            self.graphs.append(Mi)

            # set values for next iteration
            M = Mi
            k = new_k
            points = new_points
            indicies = new_topics_indicies

            iterations += 1

        return cliques
