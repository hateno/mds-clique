import numpy as np
import sklearn.manifold.rmds as rmds
import logging, math, os

import sim.topics, tvconf

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_random_state

logger = logging.getLogger()

class Relative:
    '''
    Own implementation of relative MDS adapted from sklearn 0.19.X MDS
    '''
    def __init__(self, dist_matrix, clusters, rdim=2, basis=None, basis_points=None):
        self.basis_points = basis_points
        self.dist_matrix = dist_matrix
        self.clusters = clusters
        self.rdim = rdim

        # select and project basis vector set
        self.vector_ids = basis
        if basis is None:
            self.vector_ids = self._select_basis_vectors()
        self.basis_dist_matrix = self._get_basis_matrix()
        if basis_points is None:
            self.basis_points = self._project_basis_vectors()

        # relative mapping
        self.rmds = rmds.RMDS(n_components=rdim, n_jobs=1, dissimilarity='precomputed', random_state=tvconf.SEED,
                verbose=0, fixed_points=self.basis_points, fixed_ids=self.vector_ids)
        self.points = self.rmds.fit_transform(dist_matrix)

        # manual stress calculation (check vs. (self.rmds.stress_ + self.basis_stress))
        self._calc_final_stress()

    @classmethod
    def get_all_points(self, vector_ids, basis_points, remaining_points):
        '''
        combines basis_points and remaining_points into single vector with positions preserved
        '''
        offset = sorted(vector_ids)
        vector_ids = [vector_id - offset.index(vector_id) for vector_id in vector_ids]
        points = np.insert(remaining_points, vector_ids, basis_points, axis=0)
        return points

    @classmethod
    def calc_stress(self, points, dissim):
        '''
        uses stress-1 function (p42 borg)
        '''
        dis = euclidean_distances(points)
        num = ((dis.ravel() - dissim.ravel()) ** 2)
        den = (dis.ravel() ** 2)
        stress = np.nansum(num / den)
        stress = math.sqrt(stress)
        return stress

    def _calc_final_stress(self):
        '''
        calculate final stress of all euclidean points (rmds result) vs disparities (actual distances)
        this is to compare against the stress we calculate from the base components
        '''
        points = self.points
        self.final_stress = self.calc_stress(points, self.dist_matrix)

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
        points = sim.topics.mds_helper(self.basis_dist_matrix, rdim=self.rdim, verbose=0)
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

    @classmethod
    def filter_dist_matrix(self, dist_matrix, vector_ids):
        '''
        remove basis_vectors from dist_matrix, get the values that will move
        '''
        rows = np.delete(dist_matrix, vector_ids, 0)
        cols = np.delete(rows, vector_ids, 1)
        return cols

    def get_basis_matrix(self):
        return self.basis_dist_matrix

    def print_basis_points(self, filename=None, subdir=None):
        path = 'out/experiment'
        if subdir is not None:
            path = '%s/%s' % (path, subdir)
            if not os.path.exists(path):
                logger.info('path %s does not exist, creating...' % path)
                os.mkdir(path)
        if filename != None:
            path = '%s/%s_basis_points' % (path, filename)
        else:
            path = '%s/basis_points' % path
        with open(path, 'w') as f:
            for idx, point in enumerate(self.basis_points):
                point_x = point[0]
                point_y = point[1]
                topic = self.vector_ids[idx]
                cluster = self.clusters[idx]
                f.write('%s %s %s %s\n' % (topic, cluster, point_x, point_y))

    def print_remaining_points(self, filename=None, subdir=None):
        path = 'out/experiment'
        if subdir is not None:
            path = '%s/%s' % (path, subdir)
            if not os.path.exists(path):
                logger.info('path %s does not exist, creating...' % path)
                os.mkdir(path)
        if filename != None:
            path = '%s/%s_relative' % (path, filename)
        with open(path, 'w') as f:
            for idx, point in enumerate(self.remaining_points):
                point_x = point[0]
                point_y = point[1]
                topic = self.dist_index[idx]
                cluster = self.clusters[idx]
                f.write('%s %s %s %s\n' % (topic, cluster, point_x, point_y))

    def print_points(self, filename=None, subdir=None, topic_indices=None):
        path = 'out/experiment'
        if subdir is not None:
            path = '%s/%s' % (path, subdir)
            if not os.path.exists(path):
                logger.info('path %s does not exist, creating...' % path)
                try:
                    os.mkdir(path)
                except FileExistsError:
                    logger.error('Unable to create directory %s' % path)
                    pass
        if filename != None:
            path = '%s/%s_relative' % (path, filename)
        else:
            path = '%s/relative' % path
        with open(path, 'w') as f:
            logging.info('writing relative.print_points to %s' % path)
            for idx, point in enumerate(self.points):
                point_x = point[0]
                point_y = point[1]
                topic = idx
                if topic_indices is not None:
                    topic = topic_indices[idx]
                cluster = self.clusters[idx]
                f.write('%s %s %s %s\n' % (topic, cluster, point_x, point_y))
