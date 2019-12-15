
# flair: https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md

# msft nlp-recipes: https://github.com/microsoft/nlp-recipes

# ======================================================================================================================

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


