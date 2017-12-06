import logging, random
import numpy as np

import sim.topics, sim.relative

logger = logging.getLogger()

class Online:
    def __init__(self, topic_dist_values, dist_matrix, clusters, index, rdim=2):
        assert(len(dist_matrix) > 2) # at least N=3 to properly perform this algorithm
        self.topic_dist_values = topic_dist_values
        self.dist_matrix = dist_matrix
        self.clusters = clusters
        self.index = index
        self.rdim = rdim
        self.n = len(topic_dist_values)

        self.control_points = sim.topics.mds_helper(dist_matrix, rdim=self.rdim, n_jobs=1)
        self.control_stress = sim.relative.Relative.calc_stress(self.control_points, dist_matrix)

    def process_batch(self, new_topic_dist_values, new_clusters, new_index):
        '''
        plot new_values using relative mds and control_points as fixed_points
        '''
        basis_length = len(self.topic_dist_values)
        combined_clusters = self.clusters
        combined_topic_dist_values = self.topic_dist_values
        combined_index = self.index
        basis_points = self.control_points
        for i, new_topic_dist_value in enumerate(new_topic_dist_values):
            logger.info('processing point %d (%d/%d)' % (new_index[i], i+1, len(new_topic_dist_values)))
            combined_topic_dist_values = np.append(combined_topic_dist_values, [new_topic_dist_value], axis=0)
            combined_index = np.append(combined_index, [new_index[i]])
            combined_dist_matrix = sim.topics.dissim_single(combined_topic_dist_values, index=combined_index)
            basis_ids = list(range(basis_length + i))
            combined_clusters = np.append(combined_clusters, new_clusters[i])

            relative = sim.relative.Relative(combined_dist_matrix, combined_clusters, basis=basis_ids, basis_points=basis_points)
            logger.info('current stress: %.4f' % relative.final_stress)
            basis_points = relative.points
            #relative.print_points(subdir=i, topic_indices=np.concatenate([self.index, new_index]))

        final_stress = relative.final_stress
        final_points = relative.points
        return (final_stress, final_points)

    def run(self):
        '''
        TODO refactor this out into sim/util.py
        '''
        all_stress = []
        basis_ids = []
        clusters = []
        population = list(range(len(self.topic_dist_values)))
        random_indicies = random.sample(population, self.n)

        for _ in range(2):
            random_index = random_indicies.pop()
            basis_ids.append(random_index)
            clusters.append(self.clusters[random_index])
        new_topic_dist_values = np.take(self.topic_dist_values, basis_ids, axis=0)
        new_dist_matrix = sim.topics.dissim_single(new_topic_dist_values, index=basis_ids)
        fixed_points = sim.topics.mds_helper(new_dist_matrix, rdim=self.rdim, n_jobs=1)
        all_stress.append(sim.relative.Relative.calc_stress(fixed_points, new_dist_matrix))

        while len(random_indicies) > 0:
            new_point_index = random_indicies.pop()
            clusters.append(self.clusters[new_point_index])
            logger.info('%d points left...' % (self.n - len(basis_ids)))

            # compute new dist_matrix with basis_ids and new_point_index
            new_index = np.concatenate([basis_ids, [new_point_index]])
            new_topic_dist_values = np.take(self.topic_dist_values, new_index, axis=0)
            new_dist_matrix = np.array(sim.topics.dissim_single(new_topic_dist_values, index=new_index))

            relative = sim.relative.Relative(new_dist_matrix, clusters, basis_points=fixed_points)
            new_point = relative.points[-1]

            basis_ids.append(new_point_index) # update number of fixed points
            fixed_points = np.append(fixed_points, [new_point], axis=0)
            all_stress.append(relative.final_stress)

        self.final_points = relative.points
        self.relative = relative
        self.online_points = self._rearrange_points(basis_ids)
        return all_stress

    def _rearrange_points(self, basis_ids):
        online_points = np.zeros((self.n, self.rdim))
        for idx, basis_id in enumerate(basis_ids):
            online_points[basis_id][0] = self.relative.points[idx][0]
            online_points[basis_id][1] = self.relative.points[idx][1]
        return online_points
