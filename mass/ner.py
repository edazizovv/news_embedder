import json
import pandas
import subprocess
from configuration import form_factor

opened = './data/source.xlsx'
closed = './data/gained.xlsx'


# flair
def flair_ner_cell(data, config):
    data.to_excel(opened, index=False)
    form = form_factor(config)
    with open('./data/params.json', 'w') as js:
        json.dump(form, js)
    subprocess.call([config.virtual.flair, './mass/low_level/ner_flair.py'])
    data = pandas.read_excel(closed)
    return data


# deeppavlov
def deeppavlov_ner_cell(data, config):
    data.to_excel(opened, index=False)
    form = form_factor(config)
    with open('./data/params.json', 'w') as js:
        json.dump(form, js)
    subprocess.call([config.virtual.deeppavlov, './mass/low_level/ner_deeppavlov.py'])
    data = pandas.read_excel(closed)
    return data


# spacy
def spacy_ner_cell(data, config):
    data.to_excel(opened, index=False)
    form = form_factor(config)
    with open('./data/params.json', 'w') as js:
        json.dump(form, js)
    subprocess.call([config.virtual.spacy, './mass/low_level/ner_spacy.py'])
    data = pandas.read_excel(closed)
    return data


# nltk stanford
def nltk_stanford_ner_cell(data, config):
    data.to_excel(opened, index=False)
    form = form_factor(config)
    with open('./data/params.json', 'w') as js:
        json.dump(form, js)
    subprocess.call([config.virtual.nltk, './mass/low_level/ner_nltk_stanford.py'])
    data = pandas.read_excel(closed)
    return data


