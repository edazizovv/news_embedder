import numpy
import pandas
import subprocess

"""
Functional interface is the following:
input: 'array' as 'list' of 'str'; conditional '*args'
output: a dictionary with the following structure: {<sentiment code>: <sentiment value>, ...}

"""

# ----------------------------------------------------------------------------------------------------------------------
# flair

"""
def flair_assessor(array, *args):
    from flair.data import Sentence
    from flair.models import TextClassifier
    classifier = TextClassifier.load('en-sentiment')
    result = []
    columns = ['positive', 'negative']

    for x in array:
        sentence = Sentence(x)

        # predict NER tags
        predicted = classifier.predict(sentence)

        values = None
        if predicted[0].labels[0].value == 'NEGATIVE':
            values = numpy.array([0, predicted[0].labels[0].score])
            # score['positive'] = 0
            # score['negative'] = predicted[0].labels[0].score
        if predicted[0].labels[0].value == 'POSITIVE':
            values = numpy.array([predicted[0].labels[0].score, 0])
            # score['positive'] = predicted[0].labels[0].score
            # score['negative'] = 0
        result.append(values.reshape(1, -1))
    result = numpy.concatenate(result, axis=0)
    #return score[key]
    # return score
    return columns, result
"""

def flair_assessor(data, config, *args):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    subprocess.call([config.virtual.flair, './mass/low_level/sentiment_flair.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data

# ----------------------------------------------------------------------------------------------------------------------
# nltk

"""
def nltk_assessor(array, *args):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    result = []
    columns = ['neg', 'neu', 'pos', 'compound']

    sid = SentimentIntensityAnalyzer()

    values = None
    for x in array:
        ss = sid.polarity_scores(x)
        values = numpy.array([ss['neg'], ss['neu'], ss['pos'], ss['compound']])
        result.append(values.reshape(1, -1))
    result = numpy.concatenate(result, axis=0)
    #return ss[key]
    # return ss
    return columns, result
"""
def nltk_assessor(data, config, *args):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    resy = subprocess.call([config.virtual.nltk, './mass/low_level/sentiment_nltk.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data

# ----------------------------------------------------------------------------------------------------------------------
# textblob

"""
def textblob_assessor(array, *args):
    from textblob import TextBlob
    result = []
    columns = ['polarity', 'subjectivity']

    values = None
    for x in array:
        blob = TextBlob(x)
        values = numpy.array([blob.sentiment.polarity, blob.sentiment.subjectivity])
        result.append(values.reshape(1, -1))
    result = numpy.concatenate(result, axis=0)
    #return sss[key]
    # return sss
    return columns, result
"""
def textblob_assessor(data, config, *args):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    resy = subprocess.call([config.virtual.textblob, './mass/low_level/sentiment_textblob.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data

# ----------------------------------------------------------------------------------------------------------------------
# pattern

"""
def pattern_assessor(array, *args):
    from pattern.web import plaintext
    from pattern.en import polarity, subjectivity
    result = []
    columns = ['polarity', 'subjectivity']

    values = None
    for x in array:
        pt = plaintext(x)
        values = numpy.array([polarity(pt), subjectivity(pt)])
        result.append(values.reshape(1, -1))
    result = numpy.concatenate(result, axis=0)
    #return sss[key]
    # return sss
    return columns, result
"""

def pattern_assessor(data, config, *args):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    resy = subprocess.call([config.virtual.pattern, './mass/low_level/sentiment_pattern.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data
