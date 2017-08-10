import matplotlib.pyplot as plt
import numpy as np

from IPython import get_ipython
from sim.topics import cluster_topics

def setup():
    # ipython setup
    shell = get_ipython()
    shell.magic('%matplotlib qt5') # TODO move this to .ini

def plot(points, clusters, point_clusters=False, title=None, xlabel=None, ylabel=None):
    setup()

    if point_clusters:
        clusters = cluster_topics(10, points)

    # plot
    x = points[:,0]
    y = points[:,1]
    fig = plt.figure() # new figure window
    ax = fig.add_subplot(111)
    ax.scatter(x, y, marker='.', c=clusters, cmap='viridis')
    if title != None:
        plt.title(title)
    if xlabel != None:
        plt.xlabel(xlabel)
    if ylabel != None:
        plt.ylabel(ylabel)
    plt.grid('on')
    plt.show(block=True) # blocks in ipython, save figure, close to resume script

def plot_quads_helper(quads, points, clusters, title):
    points_quad = np.array([points[point] for point in quads])
    clusters_quad = [clusters[point] for point in quads]
    plot(points_quad, clusters_quad)

def plot_quads(points, clusters, points_stress):
    stresses = [x[1] for x in points_stress]
    std = np.std(stresses)
    mean = np.mean(stresses)

    quadfirst =     [x[0] for x in points_stress if x[1] <= (mean - std)]
    quadsecond =    [x[0] for x in points_stress if x[1] > (mean - std) and x[1] <= mean]
    quadthird =     [x[0] for x in points_stress if x[1] > mean and x[1] <= (mean + std)]
    quadfourth =    [x[0] for x in points_stress if x[1] > (mean + std)]

    # TODO update logic
    #plot_quads_helper(quadfirst, points, clusters, 'First')
    #plot_quads_helper(quadsecond, points, clusters, 'Second')
    #plot_quads_helper(quadthird, points, clusters, 'Third')
    #plot_quads_helper(quadfourth, points, clusters, 'Fourth')

    return [quadfirst, quadsecond, quadthird, quadfourth]
