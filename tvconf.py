import ConfigParser

# read configuration
config = ConfigParser.RawConfigParser()
config.read('config.ini')

CORPUS = config.get('Global', 'corpus')
STOPWORDS = config.get('Global', 'stopwords')
