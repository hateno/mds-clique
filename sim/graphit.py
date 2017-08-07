import matplotlib.pyplot as plt

from IPython import get_ipython
from sim.topics import cluster_topics

def setup():
    # ipython setup
    shell = get_ipython()
    shell.magic('%matplotlib qt5')

def plot(points, clusters, point_clusters=False):
    setup()

    if point_clusters:
        clusters = cluster_topics(10, points)

    # plot
    x = points[:,0]
    y = points[:,1]
    plt.scatter(x, y, marker='.', c=clusters, cmap='viridis')
