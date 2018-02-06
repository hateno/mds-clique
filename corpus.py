import logging, nltk, os, tvconf
from gensim import corpora
from gensim.parsing.porter import PorterStemmer
from six import iteritems
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download('stopwords')

# make store directory
if not os.path.exists('store/'):
    os.mkdir('store')

# read corpus
documents = []
for doc in os.listdir(tvconf.CORPUS):
    f = open('%s/%s' % (tvconf.CORPUS, doc), 'r')
    text = f.read()
    documents.append(text)
    f.close()

# stoplist
'''
f = open(tvconf.STOPWORDS, 'r')
stopword_file = f.read()
f.close()
'''

stoplist = set()
#for stopword in stopword_file.split('\n'):
for stopword in stopwords.words('english'):
    stoplist.add(stopword)

# remove common words and tokenize
sno = SnowballStemmer('english')
texts = []
tokenizer = RegexpTokenizer(r'\w+')
for document in documents:
    intermediate = [sno.stem(word) for word in tokenizer.tokenize(document) if word not in stoplist]
    texts.append(intermediate)

# remove words that appears only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('store/corpus.mm', corpus)

# dict
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
        if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed
dictionary.save('store/corpus.dict')
