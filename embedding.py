
# flair:            https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md
#                   https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md

# sister:           https://towardsdatascience.com/super-easy-way-to-get-sentence-embedding-using-fasttext-in-python-a70f34ac5b7c

# spaCy:            https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/

# bert-embedding:   https://pypi.org/project/bert-embedding/

# USE:              https://tfhub.dev/google/universal-sentence-encoder/4

# bert-as-service:  https://github.com/hanxiao/bert-as-service?source=post_page-----1dbfe6a66f1d----------------------


# https://www.clips.uantwerpen.be/pages/pattern-vector#tf-idf
# https://medium.com/@premrajnarkhede/sentence2vec-evaluation-of-popular-theories-part-i-simple-average-of-word-vectors-3399f1183afe


# !
# HUGE: we can grab any layers or their combinations from there!
# http://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#1-loading-pre-trained-bert
# !
# ----------------------------------------------------------------------------------------------------------------------
# flair

from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings

# CLASSIC WORD EMBEDDINGS

def flair_embeddings(x, *args):
    code = args[0]
    embedding = None
    if code == 'glove':
        embedding = WordEmbeddings('glove')
    if code == 'turian':
        embedding = WordEmbeddings('turian')
    if code == 'extvec':
        embedding = WordEmbeddings('extvec')
    if code == 'crawl':
        embedding = WordEmbeddings('crawl')
    if code == 'news':
        embedding = WordEmbeddings('news')
    if code == 'twitter':
        embedding = WordEmbeddings('twitter')
    if code == 'en-wiki':
        embedding = WordEmbeddings('en-wiki')
    if code == 'en-crawl':
        embedding = WordEmbeddings('en-crawl')
    if code == 'en-forward':
        embedding = FlairEmbeddings('en-forward')
    if code == 'en-backward':
        embedding = FlairEmbeddings('en-backward')
    if code == 'news-forward':
        embedding = FlairEmbeddings('news-forward')
    if code == 'news-backward':
        embedding = FlairEmbeddings('news-backward')
    if code == 'mix-forward':
        embedding = FlairEmbeddings('mix-forward')
    if code == 'mix-backward':
        embedding = FlairEmbeddings('mix-backward')
    if embedding is None:
        raise KeyError("Insufficient vespine gas")
    sentence = Sentence(x)
    embed = embedding.embed(sentence)
    return sentence, embed

# GLOVE embeddings
# embedding = WordEmbeddings('glove')

# TURIAN embeddings
# embedding = WordEmbeddings('turian')

# KOMNINOS embeddings
# embedding = WordEmbeddings('extvec')

# FT-CRAWL embeddings
# embedding = WordEmbeddings('crawl')

# FT-CRAWL embeddings
# embedding = WordEmbeddings('news')

# twitter embeddings
# embedding = WordEmbeddings('twitter')

# two-letter language code wiki embeddings
# embedding = WordEmbeddings('en-wiki')

# two-letter language code crawl embeddings
# embedding = WordEmbeddings('en-crawl')

# FLAIR EMBEDDINGS

# en-forward
# embedding = FlairEmbeddings('en-forward')

# en-backward
# embedding = FlairEmbeddings('en-backward')

# en-forward-fast
# embedding = FlairEmbeddings('en-backward')

# en-backward-fast
# embedding = FlairEmbeddings('en-backward')

# news-forward
# embedding = FlairEmbeddings('news-forward')

# news-backward
# embedding = FlairEmbeddings('news-backward')

# news-forward-fast
# embedding = FlairEmbeddings('news-forward-fast')

# news-backward-fast
# embedding = FlairEmbeddings('news-backward-fast')

# mix-forward
# embedding = FlairEmbeddings('mix-forward')

# mix-backward
# embedding = FlairEmbeddings('mix-backward')

"""
sentence = Sentence('The grass is green .')

# embed a sentence using glove.
embedding.embed(sentence)

# now check out the embedded tokens.
for token in sentence:
    print(token)
    print(token.embedding)
"""


# ----------------------------------------------------------------------------------------------------------------------
# sister

import sister

def sister_embeddings(x, *args):
    embedding = sister.MeanEmbedding(lang="en")
    return embedding(x)

"""
embedding = sister.MeanEmbedding(lang="en")

sentence = "I am a dog."
vector = embedding(sentence)  # 300-dim vector
"""

# ----------------------------------------------------------------------------------------------------------------------
# spaCy
import spacy
def spacy_embeddings(x, *args):
    model = args[0]
    embedder = None
    if model == 'sm':
        embedder = spacy.load('en_core_web_sm')
    if model == 'md':
        embedder = spacy.load('en_core_web_md')
    if model == 'lg':
        embedder = spacy.load('en_core_web_lg')
    if embedder is None:
        raise KeyError("Insufficient vespine gas")
    doc = embedder(x)
    return doc.vector

"""
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_md
# python -m spacy download en_core_web_lg

import spacy
# Load the spacy model that you have installed
nlp = spacy.load('en_core_web_sm')
# nlp = spacy.load('en_core_web_md')
# nlp = spacy.load('en_core_web_lg')
# process a sentence using the model
doc = nlp("This is some text that I am processing with Spacy")
embedded = doc.vector
"""

# ----------------------------------------------------------------------------------------------------------------------
# bert-embedding
from bert_embedding import BertEmbedding
def bert_embedding(x, *args):
    dataset = args[0]
    embedding = None
    if dataset == 'book_corpus_wiki_en_uncased':
        embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased')
    if dataset == 'book_corpus_wiki_en_cased':
        embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
    if dataset == 'wiki_multilingual':
        embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual')
    if dataset == 'wiki_multilingual_cased':
        embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased')
    if dataset == 'book_corpus_wiki_en_cased':
        embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
    if embedding is None:
        raise KeyError("Insufficient vespine gas")
    result = embedding(x)
    return result

"""
from bert_embedding import BertEmbedding

bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased')
# bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_cased')
# bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual')
# bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased')
# bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')

sentence = ["Elon Musk's spaceships invaded Mars, aliens' press release says"]

# looks like they yield the same results...
result = bert_embedding(sentence)
# result = bert_embedding(sentence, 'avg')  # is default?
# result = bert_embedding(sentence, 'sum')
# result = bert_embedding(sentence, 'last')
"""

# ----------------------------------------------------------------------------------------------------------------------
# Universal Sentence Encoder (USE)
import tensorflow_hub
def use_embedding(x, *args):
    embed = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    result = embed(x)
    return result

"""
import tensorflow_hub

embed = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed(["The quick brown fox jumps over the lazy dog.",
                    "I am a sentence for which I would like to get its embedding"])
"""

# ----------------------------------------------------------------------------------------------------------------------
#

# NEEDS 8GB+ RAM
# !
# bert-serving-start -model_dir C:/Sygm/RAMP/IP-02/OSTRTA/venv/news_historica/share/cased_L-12_H-768_A-12 -num_worker=4
# bert-serving-start -model_dir C:/Sygm/RAMP/IP-02/OSTRTA/venv/news_historica/share/cased_L-24_H-1024_A-16 -num_worker=4
# bert-serving-start -model_dir C:/Sygm/RAMP/IP-02/OSTRTA/venv/news_historica/share/uncased_L-12_H-768_A-12 -num_worker=4
# bert-serving-start -model_dir C:/Sygm/RAMP/IP-02/OSTRTA/venv/news_historica/share/uncased_L-24_H-1024_A-16 -num_worker=4
# !
"""
from bert_serving.client import BertClient
bc = BertClient()
bc.encode(['First do it', 'then do it right', 'then do it better'])
"""
