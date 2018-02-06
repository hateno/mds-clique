import hashlib, math, logging, pickle, os, sys
import numpy as np

import sim.random, sim.relative, sim.clique, sim.topics

from scipy.stats import sem, t
from sklearn.metrics import jaccard_similarity_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array

logger = logging.getLogger()

def preprocessing(ident):
    if not os.path.exists('out/ident'):
        os.mkdir('out/ident') # ensure out/ident exists
    ident_path = 'out/ident/%s' % ident
    if not os.path.exists(ident_path):
        os.mkdir(ident_path)

def process_stress(sample_id, basis_ids, stress, k, dim, dist_matrix, clusters):
    filename = '%d-%s' % (k, sample_id)
    subdir = '%d-all' % k
    logger.info('processing %s...' % filename)
    relative = sim.relative.Relative(dist_matrix, clusters, rdim=dim, basis=basis_ids)
    relative.print_points(filename=filename, subdir=subdir)

    stress[filename] = relative.final_stress

def calc_tval(tvalue, values):
    '''
    tvalue is int, values is a numpy array
    '''
    return tvalue * values.std() / math.sqrt(len(values))

def process_online(sample_id, args, all_stress):
    (topic_dist_values, dist_matrix, clusters) = sim.util.generate_data(n=args.n, ndim=3, n_clusters=args.clusters)
    online = sim.online.Online(topic_dist_values, dist_matrix, clusters, list(range(args.n)))
    stress = online.run()
    all_stress[sample_id] = stress

    points = sim.topics.mds_helper(dist_matrix)
    control_stress = sim.relative.Relative.calc_stress(points, dist_matrix)
    controls = list(all_stress['controls'])
    controls.append(control_stress)
    all_stress['controls'] = controls
    #sim.topics.print_mds_points(online.online_points, clusters=clusters, filename='online_points', subdir=sample_id)

def analyze_online(all_stress):
    items = all_stress.items()
    stresses = [item[1] for item in items if 'controls' != item[0]]
    df = len(stresses[0])
    tval = t.ppf(0.975, df - 1)
    avgs = [0]*df
    all_mean = []
    all_err = []
    for i in range(df):
        avgs[i] = np.array([item[i] for item in stresses])
        all_mean.append(avgs[i].mean())
        err = tval * sem(avgs[i], axis=None)
        all_err.append(err)
    logger.info('****** ANALYSIS ********')
    logger.info(' '.join(str(x) for x in all_mean))
    logger.info(' '.join(str(x) for x in all_err))
    controls = np.array(all_stress['controls'])
    logger.info('MDS: %.8f | %.8f' % (controls.mean(), tval * sem(controls, axis=None)))

def generate_data(n=100, ndim=3, n_clusters=3, ident=None, matrix=False):
    # random
    rand = sim.random.Random(n=n, ident=ident)

    if not matrix:
        # random dist values
        topic_dist_values = rand.generate_cluster_value_matrix(ndim=ndim, n_clusters=n_clusters)
        topic_dist_values = np.array(topic_dist_values)

        # dist matrix calculation
        dist_matrix_list = sim.topics.dissim(topic_dist_values)
        dist_matrix = check_array(dist_matrix_list)

        # clusters of mds
        clusters = sim.topics.cluster_topics(n_clusters, topic_dist_values)
    else:
        (topic_dist_values, dist_matrix, clusters) = rand.generate_cluster_dist_matrix(n_clusters=n_clusters)

    return (topic_dist_values, dist_matrix, clusters)

def process_relative_online_clique(experiment_id, args, final_result, ident=None):
    (topic_dist_values, dist_matrix, clusters) = generate_data(ident=ident, n=args.n, n_clusters=args.clusters)

    # data
    (ctrain, ctrain_index, ctest, ctest_index, ctrain_clusters, ctest_clusters) = sim.util.index_train_test_split(topic_dist_values, clusters)
    ctrain_dissim = sim.topics.dissim_single(ctrain, index=ctrain_index)
    ctest_dissim = sim.topics.dissim_single(ctest, index=ctest_index)

    # control
    control_online = sim.online.Online(ctrain, ctrain_dissim, ctrain_clusters, ctrain_index)
    (control_stress, control_points) = control_online.process_batch(ctest, ctest_clusters, ctest_index)

    # cliques
    cq = sim.clique.Clique(ctrain, ctrain_dissim, ctrain_clusters, args, index=ctrain_index)
    cliques = cq.find_cliques(approximation=args.approximation, parallel=args.parallel)
    n_cliques = len(cliques)
    cliques_clusters = cq.cliques_clusters

    placements = []
    stress_placements = []
    for iteration, (itest, itest_index, itest_cluster) in enumerate(zip(ctest, ctest_index, ctest_clusters)):
        #print('%s %s %s' % (str(itest), str(itest_index), str(itest_cluster)))
        min_clique, min_clusters, min_stress = cq.find_lowest_stress_clique(experiment_id, iteration, itest_index, itest_cluster, topic_dist_values)
        #print('placing point %s into %d with stress %.4f' % (str(itest_index), min_clique, min_stress))
        placements.append(min_clique)
        stress_placements.append(min_stress)

    all_cliques_stress = cq.get_cliques_stress(topic_dist_values)
    sub_control_stress = cq.get_control_stress(topic_dist_values, control_points)
    clique_points = [[item[1].tolist() for item in clique] for clique in cq.cliques]
    all_cliques_index = cq.cliques_index
    all_control_index = np.concatenate([ctrain_index, ctest_index])

    pred_clusters = [sim.topics.cluster_topics(args.clusters, clique_points) for clique_points in clique_points if len(clique_points) >= args.clusters]
    true_clusters = [np.take(np.array(clusters), index) for index in all_cliques_index if len(index) >= args.clusters]
    cluster_score = []
    for j in range(len(pred_clusters)):
        cluster_score.append(jaccard_similarity_score(true_clusters[j], pred_clusters[j]))

    final_result[experiment_id] = {'cluster_score': cluster_score, 'cliques_stress': all_cliques_stress, 'placements': placements, 'control_stress': control_stress, 'control_points': control_points, 'clique_points': clique_points, 'stress_placements': stress_placements, 'sub_control_stress': sub_control_stress, 'cliques_index': all_cliques_index, 'control_index': all_control_index}

def calc_err(values):
    values_nonan = values[~np.isnan(values)]
    df = len(values_nonan) - 1
    mvals = values_nonan.mean()
    if np.isnan(mvals):
        return 0, 0
    tval = t.ppf(0.975, df)
    err = tval * sem(values_nonan, axis=None)
    return mvals, err

def analyze(values, text):
    '''
    values in an numpy array
    '''
    mvals, err = calc_err(values)
    logger.info('%s: %.4f | %.4f' % (text, mvals, err))

def output_final_result(final_result, args, n_experiments):
    with open('out/experiment/final_result_n%d_c%d_e%d' % (args.n, args.clusters, n_experiments), 'wb') as f:
        pickle.dump(final_result.items(), f)

def analyze_relativeonline(final_result, args, n_experiments):
    logger.info('******* ANALYSIS ********')

    # average control stress
    items = final_result.items()
    all_control_stress = np.array([item[1]['control_stress'] for item in items])
    analyze(all_control_stress, 'control_stress')

    all_cliques_stress = np.concatenate([item[1]['cliques_stress'] for item in items])
    analyze(all_cliques_stress, 'cliques_stress')

    clustersall = np.concatenate([item[1]['cluster_score'] for item in items])
    analyze(clustersall, 'cluster-flat')

    all_avg_control_stress = np.array([np.array(item[1]['sub_control_stress']).mean() for item in items])
    analyze(all_avg_control_stress, 'avg_control_stress')

    all_avg_cliques_stress = np.array([np.array(item[1]['cliques_stress']).mean() for item in items])
    analyze(all_avg_cliques_stress, 'avg_cliques_stress')

    clusters = np.array([np.array(item[1]['cluster_score']).mean() for item in items])
    analyze(clusters, 'cluster-oavg')

    lengths = np.array([len(item[1]['cliques_stress']) for item in items])
    analyze(lengths, 'lengths')

    output_final_result(final_result, args, n_experiments)

def process_online_clique(experiment_id, args, final_result, dim=2):
    (topic_dist_values, dist_matrix, clusters) = generate_data(n=args.n, n_clusters=args.clusters)

    # data
    (ctrain, ctrain_index, ctest, ctest_index, ctrain_clusters, ctest_clusters) = sim.util.index_train_test_split(topic_dist_values, clusters)
    ctrain_dissim = sim.topics.dissim_single(ctrain, index=ctrain_index)
    ctest_dissim = sim.topics.dissim_single(ctest, index=ctest_index)
    # cliques
    cq = sim.clique.Clique(ctrain, ctrain_dissim, ctrain_clusters, args, index=ctrain_index)
    cliques = cq.find_cliques(approximation=args.approximation, parallel=args.parallel)
    n_cliques = len(cliques)
    cliques_clusters = cq.cliques_clusters
    # placements
    placements = []
    stress_placements = []
    for iteration, (itest, itest_index, itest_cluster) in enumerate(zip(ctest, ctest_index, ctest_clusters)):
        #print('%s %s %s' % (str(itest), str(itest_index), str(itest_cluster)))
        min_clique, min_clusters, min_stress = cq.find_lowest_stress_clique(experiment_id, iteration, itest_index, itest_cluster, topic_dist_values)
        #print('placing point %s into %d with stress %.4f' % (str(itest_index), min_clique, min_stress))
        placements.append(min_clique)
        stress_placements.append(min_stress)
    # analysis
    all_cliques_stress = cq.get_cliques_stress(topic_dist_values)
    clique_points = [[item[1].tolist() for item in clique] for clique in cq.cliques]
    cliques_index = cq.cliques_index
    pred_clusters = [sim.topics.cluster_topics(args.clusters, clique_points) for clique_points in clique_points if len(clique_points) >= args.clusters]
    true_clusters = [np.take(np.array(clusters), index) for index in cliques_index if len(index) >= args.clusters]
    cluster_score = []
    for j in range(len(pred_clusters)):
        cluster_score.append(jaccard_similarity_score(true_clusters[j], pred_clusters[j]))

    # control
    control_cq = sim.clique.Clique(topic_dist_values, dist_matrix, clusters, args, index=list(range(len(topic_dist_values))))
    control_cliques = control_cq.find_cliques(approximation=args.approximation, parallel=args.parallel)
    n_control_cliques = len(control_cliques)
    all_control_stress = control_cq.get_cliques_stress(topic_dist_values)
    #control_index = control_cq.cliques_index
    control_index = [[item[0] for item in clique] for clique in control_cliques]

    # clique score
    result_map = np.array(sim.clique.Clique.map_point_clique(cliques_index))
    control_map = np.array(sim.clique.Clique.map_point_clique(control_index))
    cscore = jaccard_similarity_score(control_map, result_map)

    final_result[experiment_id] = {'cliques_stress': all_cliques_stress, 'cluster_score': cluster_score, 'controls_stress': all_control_stress, 'clique_score': cscore}

def analyze_onlineclique(final_result, args, n_experiment):
    logger.info('******* ANALYSIS ********')

    items = final_result.items()

    scoreall = np.array([item[1]['clique_score'] for item in items]) # how online cliques match up with control group
    analyze(scoreall, 'score')

    controlall = np.concatenate([item[1]['controls_stress'] for item in items]) # all control stress
    analyze(controlall, 'control-flat')

    cliquesall = np.concatenate([item[1]['cliques_stress'] for item in items]) # all cliques stress
    analyze(cliquesall, 'cliques-flat')

    control = np.array([np.array(item[1]['controls_stress']).mean() for item in items]) # avg sub-control stress
    analyze(control, 'control-oavg')

    cliques = np.array([np.array(item[1]['cliques_stress']).mean() for item in items]) # avg cliques stress
    analyze(cliques, '*cliques-oavg')

    lengths = np.array([len(item[1]['cliques_stress']) for item in items])
    analyze(lengths, '*clique-lengths')

    lengths = np.array([len(item[1]['controls_stress']) for item in items])
    analyze(lengths, 'control-lengths')

    # calculate avg stress per clique
    max_clique_len = max([len(t[1]['cliques_stress']) for t in items])
    icliquestress = []
    ecliquestress = []
    icontrolstress = []
    econtrolstress = []
    iclusterscore = []
    eclusterscore = []
    for clique_len in range(max_clique_len):
        this_clique_stress = np.array([t[1]['cliques_stress'][clique_len] for t in items if len(t[1]['cliques_stress']) > clique_len])
        m, e = calc_err(this_clique_stress)
        icliquestress.append(m)
        ecliquestress.append(e)
        text = '\tavg-clique-stress-%d' % clique_len
        analyze(this_clique_stress, text)
        re_control_stress = np.array([t[1]['controls_stress'][clique_len] for t in items if len(t[1]['controls_stress']) > clique_len])
        m, e = calc_err(re_control_stress)
        icontrolstress.append(m)
        econtrolstress.append(e)
        analyze(re_control_stress, '\t\tre-control-stress-%d' % clique_len)
        this_cluster_score = np.array([t[1]['cluster_score'][clique_len] for t in items if len(t[1]['cluster_score']) > clique_len])
        m, e = calc_err(this_cluster_score)
        iclusterscore.append(m)
        eclusterscore.append(e)
        analyze(this_cluster_score, '\t\tcluster_score-%d' % clique_len)

    logger.info('*** clique stress ***')
    logger.info(' '.join(str(x) for x in icliquestress))
    logger.info(' '.join(str(x) for x in ecliquestress))
    logger.info('*** control stress ***')
    logger.info(' '.join(str(x) for x in icontrolstress))
    logger.info(' '.join(str(x) for x in econtrolstress))
    logger.info('*** cluster score ***')
    logger.info(' '.join(str(x) for x in iclusterscore))
    logger.info(' '.join(str(x) for x in eclusterscore))

def process_rmds(i, args, final_result):
    (topic_dist_values, dist_matrix, clusters) = sim.util.generate_data(n=args.n, ndim=3, n_clusters=args.clusters, matrix=args.matrix)

    # data
    (ctrain, ctrain_index, ctest, ctest_index, ctrain_clusters, ctest_clusters) = sim.util.index_train_test_split(topic_dist_values, clusters)
    ctrain_dissim = sim.topics.dissim_single(ctrain, index=ctrain_index)
    ctest_dissim = sim.topics.dissim_single(ctest, index=ctest_index)

    # control
    relative = sim.relative.Relative(dist_matrix, clusters, basis=ctrain_index)
    control_stress = relative.final_stress

    # cliques
    cq = sim.clique.Clique(ctrain, ctrain_dissim, ctrain_clusters, args, index=ctrain_index)
    cliques = cq.find_cliques(approximation=args.approximation, parallel=args.parallel)
    cliques_index = cq.cliques_index
    sub_control_stress = cq.get_control_stress(topic_dist_values, relative.points)

    # place into each clique
    cliques_stress = []
    for clique_index, clique in zip(cliques_index, cliques):
        sub_train_index = list(set(ctrain_index) - set(clique_index))
        this_index = clique_index + list(ctest_index) + sub_train_index
        clique_dist_matrix = np.take(dist_matrix, this_index, axis=1)
        clique_dist_matrix = np.take(clique_dist_matrix, this_index, axis=0)
        clusters = [0]*len(this_index)
        #basis = ctest_index - len(this_index)
        offset = sorted(this_index)
        basis = [offset.index(index) for index in list(ctest_index) + list(sub_train_index)]
        relative = sim.relative.Relative(clique_dist_matrix, clusters, basis=basis)
        cliques_stress.append(relative.final_stress)

    # control-cliques, regular mds on points from each clique
    re_control_stress = []
    for clique_index in cliques_index:
        re_topic_dist_values = np.take(topic_dist_values, clique_index, axis=0)
        re_dissim = sim.topics.dissim(re_topic_dist_values, index=clique_index)
        re_control_points = sim.topics.mds_helper(re_dissim)
        re_stress = sim.relative.Relative.calc_stress(re_control_points, re_dissim)
        re_control_stress.append(re_stress)

    final_result[i] = {'control_stress': control_stress, 'cliques_stress': cliques_stress, 're_control_stress': re_control_stress, 'sub_control_stress': sub_control_stress}

def analyze_rmds(final_result, args, n_experiments):
    logger.info('******* ANALYSIS ********')

    items = final_result.items()

    controlall = np.array([item[1]['control_stress'] for item in items]) # all control stress
    analyze(controlall, 'control-flat')

    cliquesall = np.concatenate([item[1]['cliques_stress'] for item in items]) # all cliques stress
    analyze(cliquesall, 'cliques-flat')

    control = np.array([np.nanmean(np.array(item[1]['sub_control_stress'])) for item in items]) # avg sub-control stress
    analyze(control, 'control-oavg')

    cliques = np.array([np.nanmean(np.array(item[1]['cliques_stress'])) for item in items]) # avg cliques stress
    analyze(cliques, '*cliques-oavg')

    max_clique_len = max([len(t[1]['cliques_stress']) for t in items])
    icliquestress = []
    ecliquestress = []
    icontrolstress = []
    econtrolstress = []
    for clique_len in range(max_clique_len):
        this_clique_stress = np.array([t[1]['cliques_stress'][clique_len] for t in items if len(t[1]['cliques_stress']) > clique_len])
        m, e = calc_err(this_clique_stress)
        icliquestress.append(m)
        ecliquestress.append(e)
        re_control_stress = np.array([t[1]['re_control_stress'][clique_len] for t in items if len(t[1]['re_control_stress']) > clique_len])
        m, e = calc_err(re_control_stress)
        icontrolstress.append(m)
        econtrolstress.append(e)
        #analyze(re_control_stress, '\t\tre-control-stress-%d' % clique_len)

    logger.info('*** clique stress ***')
    logger.info(' '.join(str(x) for x in icliquestress))
    logger.info(' '.join(str(x) for x in ecliquestress))
    logger.info('*** control stress ***')
    #logger.info(' '.join(str(x) for x in icontrolstress))
    #logger.info(' '.join(str(x) for x in econtrolstress))
    #logger.info(' '.join(str(x) for x in controlall))

def process_rclique(i, args, final_result):
    (topic_dist_values, dist_matrix, clusters) = sim.util.generate_data(n=args.n, ndim=3, n_clusters=args.clusters, matrix=args.matrix)
    #control_points = sim.topics.mds_helper(dist_matrix)

    clique = sim.clique.Clique(topic_dist_values, dist_matrix, clusters, args, index=list(range(args.n)))
    cliques = clique.find_cliques(approximation=args.approximation)

    control_points = clique.points
    control_stress = sim.relative.Relative.calc_stress(control_points, dist_matrix)
    control_aspe = sim.topics.calc_aspe(control_stress, args.n) #args.n * (args.n - 1) / 2)

    #sim.topics.print_mds_points(clique.points, clusters, filename='%d-mds' % i, subdir=i)

    all_cliques_points = [[item[1].tolist() for item in tupl] for tupl in cliques]
    all_cliques_stress = clique.get_cliques_stress(topic_dist_values)
    all_cliques_index = clique.cliques_index
    all_cliques_aspe = sim.topics.calc_aspe_cliques(all_cliques_stress, all_cliques_index)
    all_clique_clusters = clique.get_clique_clusters(clusters, list(range(args.n)))
    sub_control_stress = clique.get_control_stress(topic_dist_values, control_points)

    pred_clusters = [sim.topics.cluster_topics(args.clusters, clique_points) for clique_points in all_cliques_points if len(clique_points) >= args.clusters]
    true_clusters = [np.take(np.array(clusters), index) for index in all_cliques_index if len(index) >= args.clusters]
    cluster_score = []
    for j in range(len(pred_clusters)):
        cluster_score.append(jaccard_similarity_score(true_clusters[j], pred_clusters[j]))

    # control-cliques, regular mds on points from each clique
    re_control_stress = []
    for clique_index in all_cliques_index:
        re_topic_dist_values = np.take(topic_dist_values, clique_index, axis=0)
        re_dissim = sim.topics.dissim(re_topic_dist_values, index=clique_index)
        re_control_points = sim.topics.mds_helper(re_dissim)
        re_stress = sim.relative.Relative.calc_stress(re_control_points, re_dissim)
        re_control_stress.append(re_stress)

    if args.rmds:
        logging.info('Running clique with RMDS flag enabled...')
        clique.calc_rmds(subdir=i)

    #clique.print_result_points_clusters(all_cliques_points, all_cliques_index, all_clique_clusters, tag=i)
    #sim.topics.print_topic_dist_values(topic_dist_values, clusters, i)
    final_result[i] = {'cliques_stress': all_cliques_stress, 'cliques_index': all_cliques_index, 'sub_control_stress': sub_control_stress, 'control_stress': control_stress, 'cluster_score': cluster_score, 're_control_stress': re_control_stress, 'rmds_clique_stress': clique.clique_stress_values, 'control_aspe': control_aspe, 'cliques_aspe': all_cliques_aspe}

def analyze_rclique(final_result, args, n_experiments):
    logger.info('******* ANALYSIS ********')

    items = final_result.items()

    controlall = np.array([item[1]['control_stress'] for item in items]) # all control stress
    analyze(controlall, 'control-flat')

    cliquesall = np.concatenate([item[1]['cliques_stress'] for item in items]) # all cliques stress
    analyze(cliquesall, 'cliques-flat')

    #clustersall = np.concatenate([item[1]['cluster_score'] for item in items])
    #analyze(clustersall, 'cluster-flat')

    controlaspe = np.array([item[1]['control_aspe'] for item in items])
    analyze(controlaspe, 'control-aspe')

    cliqueaspe = np.concatenate([item[1]['cliques_aspe'] for item in items])
    analyze(cliqueaspe, 'cliques-aspe')

    control = np.array([np.nanmean(np.array(item[1]['sub_control_stress'])) for item in items]) # avg sub-control stress
    analyze(control, 'control-oavg')

    cliques = np.array([np.nanmean(np.array(item[1]['cliques_stress'])) for item in items]) # avg cliques stress
    analyze(cliques, '*cliques-oavg')

    clusters = np.array([np.nanmean(np.array(item[1]['cluster_score'])) for item in items])
    analyze(clusters, 'cluster-oavg')

    lengths = np.array([len(item[1]['cliques_stress']) for item in items])
    analyze(lengths, '*lengths')

    # calculate avg stress per clique
    max_clique_len = max([len(t[1]['cliques_stress']) for t in items])
    icliquestress = []
    ecliquestress = []
    icontrolstress = []
    econtrolstress = []
    irmdscliquestress = []
    ermdscliquestress = []
    ipoints = []
    epoints = []
    iaspe = []
    easpe = []
    for clique_len in range(max_clique_len):
        this_clique_stress = np.array([t[1]['cliques_stress'][clique_len] for t in items if len(t[1]['cliques_stress']) > clique_len])
        m, e = calc_err(this_clique_stress)
        icliquestress.append(m)
        ecliquestress.append(e)
        text = '\tavg-clique-stress-%d' % clique_len
        analyze(this_clique_stress, text)
        re_control_stress = np.array([t[1]['re_control_stress'][clique_len] for t in items if len(t[1]['re_control_stress']) > clique_len])
        m, e = calc_err(re_control_stress)
        icontrolstress.append(m)
        econtrolstress.append(e)
        analyze(re_control_stress, '\t\tre-control-stress-%d' % clique_len)
        rmds_clique_stress = np.array([t[1]['rmds_clique_stress'][clique_len] for t in items if len(t[1]['rmds_clique_stress']) > clique_len])
        m, e = calc_err(rmds_clique_stress)
        irmdscliquestress.append(m)
        ermdscliquestress.append(e)
        analyze(rmds_clique_stress, '\t\t\trmds-clique-stress-%d' % clique_len)
        num_points = np.array([len(t[1]['cliques_index'][clique_len]) for t in items if len(t[1]['cliques_index']) > clique_len])
        m, e = calc_err(num_points)
        ipoints.append(m)
        epoints.append(e)
        analyze(num_points, '\t\t\tnum-points-%d' % clique_len)
        aspe = np.array([t[1]['cliques_aspe'][clique_len] for t in items if len(t[1]['cliques_aspe']) > clique_len])
        analyze(aspe, '\t\taspe-%d' % clique_len)
        m, e = calc_err(aspe)
        iaspe.append(m)
        easpe.append(e)

    logger.info('*** clique stress ***')
    logger.info(' '.join(str(x) for x in icliquestress))
    logger.info(' '.join(str(x) for x in ecliquestress))
    logger.info('*** control stress ***')
    logger.info(' '.join(str(x) for x in icontrolstress))
    logger.info(' '.join(str(x) for x in econtrolstress))
    logger.info('*** rmds stress ***')
    logger.info(' '.join(str(x) for x in irmdscliquestress))
    logger.info(' '.join(str(x) for x in ermdscliquestress))
    logger.info('*** num points ***')
    logger.info(' '.join(str(x) for x in ipoints))
    logger.info(' '.join(str(x) for x in epoints))
    logger.info('*** aspe ***')
    logger.info(' '.join(str(x) for x in iaspe))
    logger.info(' '.join(str(x) for x in easpe))

    output_final_result(final_result, args, n_experiments)

def index_train_test_split(topic_dist_values, clusters):
    '''
    split topic_dist_values into test and train sets, also return original indicies as well
    '''
    index_topic_dist_values = [(i, (topic_dist_value)) for i, topic_dist_value in enumerate(topic_dist_values)]
    train, test = train_test_split(index_topic_dist_values)
    train_index = np.array([t[0] for t in train])
    train_data = np.array([t[1] for t in train])
    test_index = np.array([t[0] for t in test])
    test_data = np.array([t[1] for t in test])
    train_clusters = np.array([clusters[i] for i in train_index])
    test_clusters = np.array([clusters[i] for i in test_index])
    return train_data, train_index, test_data, test_index, train_clusters, test_clusters

def read_textfile(filepath):
    lines = []
    with open(filepath, 'r') as f:
        lines = [[float(char) for char in line.rstrip('\n').split(' ')] for line in f.readlines()]
    length = len(lines)
    n = 1
    matrix = np.zeros([length, length])
    is_triangle = False

    for line in lines: # check proper matrix
        if len(line) != length:
            if len(line) != n:
                logger.error('Import text dissimilarity failed, need to either be proper matrix or triangle matrix')
                sys.exit()
            is_triangle = True
        n += 1

    for i_row, row in enumerate(lines):
        for i_col, col in enumerate(row):
            matrix[i_row][i_col] = col
            if is_triangle:
                matrix[i_col][i_row] = col

    dummy = [(None, matrix) for _ in range(length)] # hack solution
    return (dummy, matrix)
