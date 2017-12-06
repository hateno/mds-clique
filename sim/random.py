import math, os, random, tvconf
import numpy as np
import sim.topics

class Random:
    def __init__(self, n=10, ident=None):
        self.n = n
        self.ident = ident

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

    def generate_cluster_dist_matrix(self, n_clusters=3):
        r = random.SystemRandom()
        matrix = np.zeros([self.n, self.n])
        cluster_size = self.n / n_clusters
        ceiling = cluster_size
        clusters = np.zeros([self.n])
        cluster = 1
        for row in range(self.n):
            if row > ceiling:
                ceiling += cluster_size
                cluster += 1
            matrix[row][row] = 0
            clusters[row] = cluster
            for col in range(row+1, self.n):
                distance = None
                if col <= ceiling: # short distance
                    distance = r.uniform(0.01, 0.1)
                else: # large distance
                    distance = r.uniform(1.75, 2.0)
                matrix[row][col] = distance
                matrix[col][row] = distance
        self.cluster_dist_matrix = matrix
        self.clusters = clusters
        dummy = [(None, matrix) for _ in range(self.n)] # hack solution
        '''
        with open(tvconf.MATRIX, 'wb') as f:
            np.save(f, matrix)
        '''
        return (dummy, matrix, clusters)

    def generate_cluster_value_matrix(self, ndim=3, n_clusters=3):
        '''
        randomly generates sets of spherical points, each cluster corresponds to each dimension
        regen: if False, use the previous pickled data
        '''
        if self.ident is not None:
            self.random_points_filename = 'random_%s_points_%s' % (self.n, ndim)
            points = sim.topics.pickle_load(self.ident, self.random_points_filename)
            if points is not None:
                return points

        npoints = self.n // n_clusters
        remainder = self.n % n_clusters

        col_index = 0
        points = np.zeros([0,ndim]) # data structure to concatenate generated points

        cluster = 1
        clusters = []

        for i in range(n_clusters):
            # add extra points to last cluster
            if i == n_clusters - 1:
                npoints += remainder

            # generate sphere and transform
            sphere = self.generate_spherical(npoints, ndim=ndim)
            sphere += 1
            offset = col_index // ndim # if n_clusters is higher than dimension, we want to offset to create more cluster
            sphere[:,(col_index % ndim)] += 1 + (offset * 1.5) # offsets point to be in their respective cluster
            points = np.concatenate((points, sphere))

            # clusters
            for j in range(npoints):
                clusters.append(cluster)

            # next iteration step
            cluster += 1
            col_index += 1

        if self.ident is not None:
            sim.topics.pickle_store(self.ident, self.random_points_filename, points)

        return np.array(points)

    def generate_spherical(self, npoints, ndim=3):
        r = random.SystemRandom()
        np.random.seed(r.randint(1, 2 ** 32 - 1))
        vec = np.random.randn(npoints, ndim)
        vec /= np.linalg.norm(vec, axis=0)
        return vec
