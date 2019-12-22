
# flair: https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md

# nltk: https://www.nltk.org/howto/sentiment.html

# textblob: https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis

# pattern: https://www.clips.uantwerpen.be/pages/pattern-examples-elections

# stanford corenlp: https://www.thinkinfi.com/2019/02/advanced-nlp-with-stanford-core-nlp.html

# ======================================================================================================================
# TODO: standardise interface of all following functions [1]

# CURRENT TARGET: LVL 1 FOR ALL
import pandas
# ----------------------------------------------------------------------------------------------------------------------
# flair

import numpy
from flair.data import Sentence
from flair.models import TextClassifier

classifier = TextClassifier.load('en-sentiment')


def single_estimator(x):

    sentence = Sentence(x)

    # predict NER tags
    predicted = classifier.predict(sentence)

    # print sentence with predicted labels
    score = numpy.nan
    if predicted.labels[0].value == 'NEGATIVE':
        score = -1 * predicted.labels[0].score
    if predicted.labels[0].value == 'POSITIVE':
        score = +1 * predicted.labels[0].score

    return score


def column_estimator(frame, column_name):

    frame['sentiment'] = frame[column_name].apply(func=single_estimator)
    
    return frame

# ----------------------------------------------------------------------------------------------------------------------
# nltk


from nltk.sentiment.vader import SentimentIntensityAnalyzer

def nltk_assessor(frame, column_name):

    sentences = frame[column_name].values.tolist()

    sid = SentimentIntensityAnalyzer()
    passed = []
    for j in range(len(sentences)):
        # print(sentence)
        ss = sid.polarity_scores(sentences[j])
        sss = {'n': [j]}
        sss.update({key: [ss[key]] for key in list(ss.keys())})
        passed.append(pandas.DataFrame(data=sss))
        # for k in sorted(ss):
        #    print('{0}: {1}, '.format(k, ss[k]), end='')
        # print()
    passed = pandas.concat(objs=passed, axis='index', ignore_index=True)

    return passed

# ----------------------------------------------------------------------------------------------------------------------
# textblob

from textblob import TextBlob


def textblob_assessor(frame, column_name):

    sentences = frame[column_name].values.tolist()

    blob = TextBlob(sentences)
    passed = []
    for j in range(len(sentences)):
        sss = {'n': [j]}
        sss.update({'polarity': [sentences[j].sentiment.polarity], 'subjectivity': [sentences[j].sentiment.subjectivity]})
        passed.append(pandas.DataFrame(data=sss))
    passed = pandas.concat(objs=passed, axis='index', ignore_index=True)

    return passed

# ----------------------------------------------------------------------------------------------------------------------
# pattern

from pattern.web import plaintext
from pattern.en import polarity, subjectivity


def pattern_assessor(frame, column_name):

    sentences = frame[column_name].values.tolist()

    passed = []
    for j in range(len(sentences)):
        pt = plaintext(sentences[j])
        sss = {'n': [j]}
        sss.update({'polarity': [polarity(pt)], 'subjectivity': [subjectivity(pt)]})
        passed.append(pandas.DataFrame(data=sss))
    passed = pandas.concat(objs=passed, axis='index', ignore_index=True)

    return passed

# ----------------------------------------------------------------------------------------------------------------------
# stanford corenlp


# !
# java -mx2g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
# !


from pycorenlp import StanfordCoreNLP


nlp = StanfordCoreNLP('http://localhost:9000')


def stanford_assessor(frame, column_name):

    # here it analyses every sentence -- but we need to analyse full text
    # need some aggregation?

    sentences = frame[column_name].values.tolist()

    passed = []
    for j in range(len(sentences)):
        result = nlp.annotate(sentences[j],
                              properties={
                                  'annotators': 'sentiment, ner, pos',
                                  'outputFormat': 'json',
                                  'timeout': 10000,
                              })
        s = result["sentences"][j]
        sss = {'index': [s["index"]], 'sentence': [" ".join([t["word"] for t in s["tokens"]])],
               'sentimentValue': [s["sentimentValue"]], 'sentimentSign': [s["sentiment"]]}

        passed.append(pandas.DataFrame(data=sss))

    passed = pandas.concat(objs=passed, axis='index', ignore_index=True)

    return passed

