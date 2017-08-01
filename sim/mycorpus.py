import tvconf

class MyCorpus(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __iter__(self):
        for fname in os.listdir(tvconf.CORPUS):
            for line in open('%s/%s' % (tvconf.CORPUS, fname), 'r'):
                yield self.dictionary.doc2bow(line.lower().split())
