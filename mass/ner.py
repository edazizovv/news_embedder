import json
import pandas
import subprocess


# flair
def flair_ner_cell(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    with open('./data/params.json', 'w') as param:
        json.dump(config.params, param)
    subprocess.call([config.virtual.flair, './mass/low_level/ner_flair.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


# deeppavlov
def deeppavlov_ner_cell(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    with open('./data/params.json', 'w') as param:
        json.dump(config.params, param)
    subprocess.call([config.virtual.deeppavlov, './mass/low_level/ner_deeppavlov.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


# spacy
def spacy_ner_cell(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    '''
    if config.param == 'sm':
        subprocess.call([config.virtual.spacy, './mass/low_level/ner_spacy_sm.py'])
    if config.param == 'md':
        subprocess.call([config.virtual.spacy, './mass/low_level/ner_spacy_md.py'])
    if config.param == 'lg':
        subprocess.call([config.virtual.spacy, './mass/low_level/ner_spacy_lg.py'])
    '''
    with open('./data/params.json', 'w') as param:
        json.dump(config.params, param)
    subprocess.call([config.virtual.spacy, './mass/low_level/ner_spacy.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data

# nltk stanford
def nltk_stanford_ner_cell(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    with open('./data/params.json', 'w') as param:
        json.dump(config.params, param)
    subprocess.call([config.virtual.nltk, './mass/low_level/ner_nltk_stanford.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


