import numpy as np
import random
import sim.topics

class Random:
    def __init__(self, n=10):
        self.n = n

    def generate_noisy_dist_matrix(self):
        matrix = np.empty([self.n, self.n])
        for row in range(self.n):
            matrix[row][row] = 0
            for col in range(row+1, self.n):
                distance = random.random()
                matrix[row][col] = distance
                matrix[col][row] = distance
        self.noisy_dist_matrix = matrix
        return matrix

    def generate_cluster_dist_matrix(self, num_clusters=3):
        matrix = np.zeros([self.n, self.n])
        cluster_size = self.n / num_clusters
        ceiling = cluster_size
        clusters = np.zeros([self.n])
        cluster = 1
        for row in range(self.n):
            matrix[row][row] = 0
            clusters[row] = cluster
            for col in range(row+1, self.n):
                distance = None
                if col < ceiling: # short distance
                    distance = random.uniform(0, 0.25)
                else: # large distance
                    distance = random.uniform(0.75, 1)
                matrix[row][col] = distance
                matrix[col][row] = distance
            if row > ceiling:
                ceiling += cluster_size
                cluster += 1
        self.cluster_dist_matrix = matrix
        self.clusters = clusters
        return (clusters, matrix)

    def generate_cluster_value_matrix(self, regen=True, rdim=3):
        '''
        randomly generates sets of spherical points, each cluster corresponds to each dimension
        regen: if False, use the previous pickled data
        '''
        if not regen:
            points = sim.topics.pickle_load('random_%s_points' % self.n)
            return points

        NUM_CLUSTERS = rdim # each cluster will correspond to each dimension
        npoints = self.n // NUM_CLUSTERS
        remainder = self.n % NUM_CLUSTERS

        col_index = 0
        points = np.zeros([0,rdim]) # data structure to concatenate generated points

        cluster = 1
        clusters = []

        for i in range(NUM_CLUSTERS):
            # add extra points to last cluster
            if i == NUM_CLUSTERS - 1:
                npoints += remainder

            # generate sphere and transform
            sphere = self.generate_spherical(npoints, rdim=rdim)
            sphere += 1
            sphere[:,col_index] += 1
            points = np.concatenate((points, sphere))

            # clusters
            for j in range(npoints):
                clusters.append(cluster)

            # next iteration step
            cluster += 1
            col_index += 1

        sim.topics.pickle_store('random_%s_points_d%s' % (self.n, rdim), points)

        return points

    def generate_spherical(self, npoints, rdim=3):
        vec = np.random.randn(npoints, rdim)
        vec /= np.linalg.norm(vec, axis=0)
        return vec
