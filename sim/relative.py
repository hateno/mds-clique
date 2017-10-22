import numpy as np
import sklearn.manifold.rmds as rmds

import sim.topics, tvconf

from sklearn.metrics.pairwise import euclidean_distances

class Relative:
    '''
    Own implementation of relative MDS adapted from sklearn 0.19.X MDS, general algorithm from Naud and Bernataviciene
    '''
    def __init__(self, dist_matrix, clusters, r_dim=2, basis=None):
        self.dist_matrix = dist_matrix
        self.clusters = clusters
        self.r_dim = r_dim

        # select and project basis vector set
        self.vector_ids = basis
        if basis == None:
            self.vector_ids = self._select_basis_vectors()
        self.basis_dist_matrix = self._get_basis_matrix()
        self.remaining_points = self._filter_dist_matrix()

        self.basis_points = self._project_basis_vectors()
        self.fdissim = self._get_fdissim()
        #self.basis_eucl_matrix = euclidean_distances(self.basis_points)

        # relative mapping
        self.rmds = rmds.RMDS(n_components=r_dim, n_jobs=1, dissimilarity='precomputed', random_state=tvconf.SEED,
                verbose=1, fixed_points=self.basis_points, fdissim=self.fdissim)
        result = self.rmds.fit_transform(self.remaining_points)
        self.result = result

    def _select_basis_vectors(self):
        '''
        criteria to select the initial vectors
        TODO for now, just select the first 10 vectors
        '''
        vector_ids = list(range(2))
        return vector_ids

    def _get_basis_matrix(self):
        basis_dist_matrix = np.zeros((len(self.vector_ids), len(self.vector_ids)))
        for outer_vector_id, outer_vector in enumerate(self.vector_ids):
            for inner_vector_id, inner_vector in enumerate(self.vector_ids):
                basis_dist_matrix[outer_vector_id][inner_vector_id] = self.dist_matrix[outer_vector][inner_vector]

        return basis_dist_matrix

    def _project_basis_vectors(self):
        '''
        project basis vectors using mds=2 for now
        TODO expand this to read from the program arguments
        '''
        # mds points calculation
        points = sim.topics.mds_helper(self.basis_dist_matrix, r=self.r_dim, pickle_enabled=False)
        return points

    def _get_fdissim(self):
        '''
        retrieve actual distances from the moving points to the fixed points for relative mds
        '''
        fdissim = []
        dist_index = []
        for r in range(len(self.dist_matrix)):
            if r not in self.vector_ids:
                row = []
                dist_index.append(r)
                for c in self.vector_ids:
                    distance = self.dist_matrix[r][c]
                    row.append(distance)
                fdissim.append(row)

        self.dist_index = dist_index
        return np.array(fdissim)

    def _filter_dist_matrix(self):
        '''
        remove basis_vectors from dist_matrix, get the values that will move
        '''
        rows = np.delete(self.dist_matrix, self.vector_ids, 0)
        cols = np.delete(rows, self.vector_ids, 1)
        return cols

    def get_basis_matrix(self):
        return self.basis_dist_matrix

    def print_basis_points(self, filename=None):
        path = 'out/basis_points'
        if filename != None:
            path = 'out/%s_basis_points' % filename
        with open(path, 'w') as f:
            for idx, point in enumerate(self.basis_points):
                point_x = point[0]
                point_y = point[1]
                topic = self.vector_ids[idx]
                cluster = self.clusters[idx]
                f.write('%s %s %s %s\n' % (topic, cluster, point_x, point_y))

    def print_result(self, filename=None):
        path = 'out/relative'
        if filename != None:
            path = 'out/%s_relative' % filename
        with open(path, 'w') as f:
            for idx, point in enumerate(self.result):
                point_x = point[0]
                point_y = point[1]
                topic = self.dist_index[idx]
                cluster = self.clusters[idx]
                f.write('%s %s %s %s\n' % (topic, cluster, point_x, point_y))
