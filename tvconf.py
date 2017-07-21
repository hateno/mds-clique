import configparser

# read configuration
config = configparser.RawConfigParser()
config.read('config.ini')

CORPUS = config.get('Global', 'corpus')
STOPWORDS = config.get('Global', 'stopwords')
