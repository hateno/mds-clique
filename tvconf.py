import configparser
import numpy as np

# read configuration
config = configparser.RawConfigParser()
config.read('config.ini')

CORPUS = config.get('Global', 'corpus')
STOPWORDS = config.get('Global', 'stopwords')

SEED = np.random.RandomState(seed=int(config.get('Global', 'mds_seed')))
