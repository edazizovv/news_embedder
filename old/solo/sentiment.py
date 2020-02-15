
# flair: https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md

# nltk: https://www.nltk.org/howto/sentiment.html

# textblob: https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis

# pattern: https://www.clips.uantwerpen.be/pages/pattern-examples-elections

# stanford corenlp: https://www.thinkinfi.com/2019/02/advanced-nlp-with-stanford-core-nlp.html

# ======================================================================================================================
"""
Functional interface is the following:
input: 'x' as text and '*args'
output: a dictionary with the following structure: {<sentiment code>: <sentiment value>, ...}
"""
# ======================================================================================================================
# TODO: standardise interface of all following functions [1]
# TODO: check documentation for correct use [2]
# TODO: check all warnings that are generated during runs [3]
# CURRENT TARGET: LVL 1 FOR ALL
# import pandas
# ----------------------------------------------------------------------------------------------------------------------
# flair

# import numpy




# tested
# standardised
def flair_assessor(x, *args):
    from flair.data import Sentence
    from flair.models import TextClassifier
    classifier = TextClassifier.load('en-sentiment')
    """
    x   str     A string that contains some sentences
    """
    sentence = Sentence(x)

    # predict NER tags
    predicted = classifier.predict(sentence)

    # print sentence with predicted labels
    """
    score = numpy.nan
    if predicted[0].labels[0].value == 'NEGATIVE':
        score = -1 * predicted[0].labels[0].score
    if predicted[0].labels[0].value == 'POSITIVE':
        score = +1 * predicted[0].labels[0].score
    """
    score = {}
    if predicted[0].labels[0].value == 'NEGATIVE':
        score['positive'] = 0
        score['negative'] = predicted[0].labels[0].score
    if predicted[0].labels[0].value == 'POSITIVE':
        score['positive'] = predicted[0].labels[0].score
        score['negative'] = 0
    #return score[key]
    return score



# ----------------------------------------------------------------------------------------------------------------------
# nltk


# tested
# standardised
def nltk_assessor(x, *args):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    key = args[0]
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(x)
    #return ss[key]
    return ss

# ----------------------------------------------------------------------------------------------------------------------
# textblob


# tested
# standardised
def textblob_assessor(x, *args):
    from textblob import TextBlob
    key = args[0]
    blob = TextBlob(x)
    sss = {}
    sss.update({'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity})
    #return sss[key]
    return sss

# ----------------------------------------------------------------------------------------------------------------------
# pattern


# tested
def pattern_assessor(x, *args):
    from pattern.web import plaintext
    from pattern.en import polarity, subjectivity
    key = args[0]
    pt = plaintext(x)
    sss = {}
    sss.update({'polarity': polarity(pt), 'subjectivity': subjectivity(pt)})
    #return sss[key]
    return sss

# ----------------------------------------------------------------------------------------------------------------------
# stanford corenlp

# TODO: try to make the machine run without handle start, use only python scripts [4]
# !
# cd E:\RAMP-EXTERNAL\IP-02\OSTRTA\models\stanford-corenlp-full-2018-10-05
# java -mx2g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
# !







# ? misunderstood
def stanford_assessor(x):
    from pycorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP('http://localhost:9000')
    # here it analyses every sentence -- but we need to analyse full text
    # need some aggregation?

    result = nlp.annotate(x,
                          properties={
                              # 'annotators': 'sentiment, ner, pos',
                              'annotators': 'sentiment',
                              'outputFormat': 'json',
                              'timeout': 10000,
                          })
    sss = {}
    s = result["sentences"]
    sss = {'index': [s["index"]], 'sentence': [" ".join([t["word"] for t in s["tokens"]])],
           'sentimentValue': [s["sentimentValue"]], 'sentimentSign': [s["sentiment"]]}
    return sss
"""
def stanford_assessor(frame, column_name):

    # here it analyses every sentence -- but we need to analyse full text
    # need some aggregation?

    sentences = frame[column_name].values.tolist()

    passed = []
    for j in range(len(sentences)):
        result = nlp.annotate(sentences[j],
                              properties={
                                  #'annotators': 'sentiment, ner, pos',
                                  'annotators': 'sentiment',
                                  'outputFormat': 'json',
                                  'timeout': 10000,
                              })
        s = result["sentences"][j]
        sss = {'index': [s["index"]], 'sentence': [" ".join([t["word"] for t in s["tokens"]])],
               'sentimentValue': [s["sentimentValue"]], 'sentimentSign': [s["sentiment"]]}

        passed.append(pandas.DataFrame(data=sss))

    passed = pandas.concat(objs=passed, axis='index', ignore_index=True)

    return passed
"""
