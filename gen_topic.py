import logging, os, tvconf
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionary = corpora.Dictionary.load('store/corpus.dict')

class MyCorpus(object):
    def __iter__(self):
        for fname in os.listdir(tvconf.CORPUS):
            for line in open('%s/%s' % (tvconf.CORPUS, fname), 'r'):
                yield dictionary.doc2bow(line.lower().split())

corpus = MyCorpus()

lda = models.LdaMulticore(corpus, id2word=dictionary, num_topics=100, passes=5, iterations=1500)
lda.save('store/corpus.lda')
