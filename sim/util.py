import numpy as np
from distance import Distance

# calculate dissimilarity matrix
def dissim( topics ):
    n = len( topics )
    dist = np.zeros( shape=( n, n ) )
    for i in range( n ):
        for j in range( n ):
            topic_i = topics[i]
            topic_j = topics[j]
            distance = Distance( topic_i, topic_j )
            dist[i][j] = distance.tvd()

    return dist
