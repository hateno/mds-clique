import configparser, uuid
import numpy as np

# read configuration
config = configparser.RawConfigParser()
config.read('config.ini')

CORPUS = config.get('Global', 'corpus')
STOPWORDS = config.get('Global', 'stopwords')

if 'mds_seed' in config['Global']:
    SEED = np.random.RandomState(seed=int(config.get('Global', 'mds_seed')))
else:
    SEED = np.random.RandomState()
