import numpy
import pandas
import subprocess

"""
Functional interface is the following:
input: 'array' as 'list' of 'str'; conditional '*args'
output: a dictionary with the following structure: {<sentiment code>: <sentiment value>, ...}

"""


# flair
def flair_assessor(data, config, *args):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    subprocess.call([config.virtual.flair, './mass/low_level/sentiment_flair.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


# nltk
def nltk_assessor(data, config, *args):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    resy = subprocess.call([config.virtual.nltk, './mass/low_level/sentiment_nltk.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data

# textblob
def textblob_assessor(data, config, *args):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    resy = subprocess.call([config.virtual.textblob, './mass/low_level/sentiment_textblob.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data

# pattern
def pattern_assessor(data, config, *args):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    resy = subprocess.call([config.virtual.pattern, './mass/low_level/sentiment_pattern.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data
