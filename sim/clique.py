import logging, os
import numpy as np

import sim.graph, sim.relative, sim.topics

from multiprocessing.pool import ThreadPool
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger()

class Clique:
    '''
    Run RMDS with each MDS-clique as fixed points (basis vectors), find out which clique does the document best fit into
    '''
    def __init__(self, topic_dist_values, dist_matrix, clusters, args, rdim=2, ident=None, index=None):
        self.clusters = clusters
        self.rdim = rdim
        self.dist_matrix = dist_matrix
        self.topic_dist_values = topic_dist_values
        self.ident = ident
        self.index = index
        self.clique_stress_values = []
        self.args = args
        self.N_PROCESS = mp.cpu_count() // 4
        if ident is not None:
            self.ident = '%s_clique' % ident

        self.points = sim.topics.mds_helper(dist_matrix, rdim=rdim, ident=ident)
        self.eucl_matrix = euclidean_distances(self.points)

    def find_lowest_stress_clique(self, experiment_id, iteration, newpoint_index, newpoint_cluster, master_topic_dist_values):
        stress = [] # fill out stress
        tmp_cliques = []
        tmp_dissims = []
        tmp_clusters = []
        tmp_relatives = []
        for clique, clique_cluster, clique_index in zip(self.cliques, self.cliques_clusters, self.cliques_index):
            fixed_points = np.array([item[1].tolist() for item in clique])
            fixed_ids = list(range(len(fixed_points)))
            this_cluster = clique_cluster.copy()
            this_cluster.append(newpoint_cluster)

            newpoints = self.add_newpoint(clique_index, newpoint_index, master_topic_dist_values)
            newdissim = sim.topics.dissim(newpoints, index=clique_index + newpoint_index)

            relative = sim.relative.Relative(newdissim, this_cluster, basis=fixed_ids, basis_points=fixed_points)
            stress.append(relative.final_stress)

            this_clique = clique.copy()
            this_clique.append((newpoint_index, relative.points[-1])) # store original index, and newpoint projected pair onto this clique
            tmp_cliques.append(this_clique)
            tmp_dissims.append(newdissim)
            tmp_clusters.append(this_cluster)
            tmp_relatives.append(relative)

        min_stress = min(stress)
        min_clique = stress.index(min_stress)

        # update the clique
        self.cliques[min_clique] = tmp_cliques[min_clique]
        self.cliques_clusters[min_clique] = tmp_clusters[min_clique]
        self.cliques_index[min_clique].append(newpoint_index)
        self.cliques_dissim[min_clique] = tmp_dissims[min_clique]
        #tmp_relatives[min_clique].print_points(subdir=experiment_id, filename='%d_%d' % (iteration, min_clique), topic_indices=self.cliques_index[min_clique])

        return (min_clique, self.cliques_clusters[min_clique], min_stress)

    def get_cliques_stress(self, master_topic_dist_values):
        all_clique_stress = []
        for clique, clique_index in zip(self.cliques, self.cliques_index):
            if len(clique_index) == 1: # usually occurs in approximate clique case
                all_clique_stress.append(np.nan)
            elif len(clique_index) > 1:
                clique_points = np.take(master_topic_dist_values, clique_index, axis=0)
                clique_dissim = sim.topics.dissim(clique_points, index=clique_index)
                clique_coords = np.array([item[1].tolist() for item in clique])
                clique_stress = sim.relative.Relative.calc_stress(clique_coords, clique_dissim)
                all_clique_stress.append(clique_stress)
        return all_clique_stress

    def get_control_stress(self, master_topic_dist_values, control_points):
        '''
        retrieve stress comparison that matches up with the points in the cliques
        '''
        all_control_stress = []
        for clique_index in self.cliques_index:
            if len(clique_index) > 1:
                current_points = np.take(master_topic_dist_values, clique_index, axis=0)
                control_dissim = sim.topics.dissim(current_points, index=clique_index)
                control_coords = np.take(control_points, clique_index, axis=0)
                control_stress = sim.relative.Relative.calc_stress(control_coords, control_dissim)
                all_control_stress.append(control_stress)
        return all_control_stress

    def find_cliques(self, k=0.1, approximation=False, parallel=False):
        logger.info('Running with k %.2f...' % k)
        filename = 'cliques_%s' % str(k)
        if self.ident is not None:
            self.cliques = sim.topics.pickle_load(self.ident, filename)
            if self.cliques is not None:
                return self.cliques

        self.mgraph = sim.graph.MGraph(self.points, self.topic_dist_values, self.dist_matrix, self.eucl_matrix, self.rdim, self.clusters, approximation=approximation, parallel=parallel)
        if self.args.clique == 'distance':
            logger.info('Using find_cliques_distance...')
            self.cliques = self.mgraph.find_cliques_distance(k, combine=self.args.combine)
        else:
            logger.info('Using find_cliques_stress...')
            self.cliques = self.mgraph.find_cliques_stress(combine=self.args.combine)

        self.set_clique_clusters()
        self.cliques_dissim = self.get_cliques_dissim()

        if self.index is not None: # reindex cliques
            self.cliques = self.reindex_cliques(self.cliques, self.index)
        self.set_clique_index()

        if self.ident is not None:
            sim.topics.pickle_store(self.ident, filename, self.cliques)

        return self.cliques

    def add_newpoint(self, clique_index, newpoint_index, master_topic_dist_values):
        '''
        combine the clique and newpoint from master_topic_dist_values into one
        '''
        all_index = clique_index.copy()
        all_index.append(newpoint_index)
        newpoints = np.take(master_topic_dist_values, all_index, axis=0)
        return newpoints

    def set_clique_index(self):
        self.cliques_index = []
        for clique in self.cliques:
            clique_index = [item[0] for item in clique]
            self.cliques_index.append(clique_index)

    def set_clique_clusters(self):
        self.cliques_clusters = []
        for clique in self.cliques:
            clique_clusters = []
            for item in clique:
                topic = item[0]
                cluster = self.clusters[topic]
                clique_clusters.append(cluster)
            self.cliques_clusters.append(clique_clusters)

    def reindex_cliques(self, cliques, index):
        '''
        reindex cliques topics according to the index, this is normally done to calibrate with the master topic_dist_values instead of the local version
        '''
        rcliques = []
        for clique in cliques:
            rclique = [None]*len(clique)
            pool = ThreadPool(self.N_PROCESS)
            pargs = [(i, item, rclique, index) for i, item in enumerate(clique)]
            pool.starmap(self._reindex_clique, pargs)
            pool.close()
            pool.join()
            rcliques.append(rclique)
        return rcliques

    def _reindex_clique(self, i, item, rclique, index):
        ritem = (index[item[0]], item[1])
        rclique[i] = ritem

    def get_cliques_dissim(self):
        cliques_dissim = self.calc_cliques_dissim(self.topic_dist_values, self.cliques, self.index)
        return cliques_dissim

    def calc_cliques_dissim(self, topic_dist_values, cliques, master_index):
        '''
        convert cliques into dissims
        '''
        cliques_dissim = []
        for clique in cliques:
            points = []
            master_idxs = []
            for item in clique:
                point_idx = item[0]
                master_idx = master_index[point_idx]
                master_idxs.append(master_idx)
                point = topic_dist_values[point_idx]
                points.append(point)
            clique_dissim = sim.topics.dissim(points, master_idx)
            cliques_dissim.append(clique_dissim)
        return cliques_dissim

    def process_batch_online(self, all_clique_points, all_clique_clusters, all_clique_indices, newdata, newdata_clusters, newdata_index):
        '''
        newdata: new points that will be put into self.cliques
        '''
        latest_clique_dissim = [sim.topics.dissim_single(clique, index=clique_index) for clique, clique_index in zip(all_clique_points, all_clique_indices)]
        for i, newpoint in enumerate(newdata):
            logger.info('processing point %d/%d...' % (i, len(newdata)))

            newpoint_cluster = newdata_clusters[0] # extract newpoint's cluster
            newdata_clusters = np.delete(newdata_clusters, 0) # update newdata_clusters

            all_clique_stress = [] # keep track of each clique's stress
            all_clique_dissim = [] # keep track of each clique's dissim

            for j, clique_points in enumerate(all_clique_points): # fit new point into each clique, record stress
                basis_vector = list(range(len(clique_points)))
                combined_dist_matrix = self.get_combined_dist_matrix(clique_points, [newpoint], all_clique_indices + newdata_index[i])
                clique_cluster = all_clique_clusters[j]
                combined_clusters = np.concatenate([clique_cluster, [newpoint_cluster]])
                relative = sim.relative.Relative(combined_dist_matrix, combined_clusters, basis=basis_vector)

                all_clique_stress.append(relative.final_stress)
                all_clique_dissim.append(combined_dist_matrix)

            # add new point into clique with lowest stress
            chosen_clique = all_clique_stress.index(min(all_clique_stress))
            logger.info('adding point to clique %d' % chosen_clique)

            all_clique_points[chosen_clique].append(newpoint.tolist()) # add new point to clique with lowest stress
            all_clique_clusters[chosen_clique].append(newpoint_cluster) # also update cluster for that clique
            all_clique_indices[chosen_clique].append(newdata_index[i])
            latest_clique_dissim[chosen_clique] = all_clique_dissim[chosen_clique]

        return all_clique_points, all_clique_clusters, all_clique_indices, latest_clique_dissim

    @classmethod
    def map_point_clique(self, index):
        '''
        maps a point to which clique it belongs in
        '''
        cindex = np.concatenate(index)
        n = len(cindex)
        mapping = [-1]*n
        for idx in cindex:
            mapping[idx] = [i for i, clique in enumerate(index) if idx in clique][0]
        return mapping

    def print_result_points_clusters(self, result_points, result_indices, result_clusters, tag=None):
        for i, result_point in enumerate(result_points):
            result_cluster = result_clusters[i]
            result_index = result_indices[i]
            _tag = i if tag is None else tag
            sim.topics.print_mds_points(result_point, clusters=result_cluster, filename='points_%d' % i, subdir=_tag, index=result_index)

    def get_clique_clusters(self, clusters, train_index):
        all_clique_clusters = []
        for clique in self.cliques:
            clique_indicies = [t[0] for t in clique]
            clique_clusters = [clusters[train_index[clique_index]] for clique_index in clique_indicies]
            all_clique_clusters.append(clique_clusters)
        return all_clique_clusters

    def get_combined_dist_matrix(self, points, new_points, new_index):
        '''
        combine points and new_points and get their combined dist_matrix
        '''
        combined_points = np.concatenate([points, new_points])
        combined_dist_matrix = sim.topics.dissim(combined_points, index=new_index)
        return combined_dist_matrix

    def get_clique_points(self, topic_dist_values, train_index):
        '''
        take points from a clique, map it to train_index and to retrieve original topic_dist_value point
        '''
        all_clique_points = []
        all_clique_indicies = []
        for clique in self.cliques:
            clique_indicies = [train_index[t[0]] for t in clique]
            all_clique_indicies.append(clique_indicies)
            clique_points = [topic_dist_values[clique_index].tolist() for clique_index in clique_indicies]
            all_clique_points.append(clique_points)
        return all_clique_points, all_clique_indicies

    def calc_rmds(self, subdir=None):
        self._calc_basis_vectors()
        self.clique_stress_values = []

        relative_instances = []
        for idx, basis_vector in enumerate(self.basis_vectors_list):
            logger.info('%s' % basis_vector)
            relative = sim.relative.Relative(self.dist_matrix, self.clusters, basis=basis_vector)
            relative_instances.append(relative) # store rmds instances for reference and debugging

            #filename = 'clique-%d' % idx
            #relative.print_basis_points(filename=filename, subdir=subdir)
            #relative.print_points(filename=filename, subdir=subdir)
            clique_stress_value = relative.final_stress
            self.clique_stress_values.append(clique_stress_value)

        self.relative_instances = relative_instances

    def _calc_basis_vectors(self):
        basis_vectors_list = []
        for clique in self.cliques:
            basis_vector = [tupl[0] for tupl in clique]
            basis_vectors_list.append(basis_vector)

        self.basis_vectors_list = basis_vectors_list

    @classmethod
    def output_clique_lengths(self, all_points, filetag):
        '''
        currently used with single clique
        '''
        item_cliques = all_points.items()
        item_cliques = sorted(item_cliques, key=lambda t: t[0])
        with open('out/num_clique_points_%s' % filetag, 'w') as f:
            for item_clique in item_cliques:
                for clique in item_clique[1]:
                    f.write('%d ' % len(clique['basis_points']))
                f.write('\n')
