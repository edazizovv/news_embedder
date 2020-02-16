import os
import json
import pandas
import subprocess
from news_embedder.configuration import form_factor

current_wd = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(__file__))
params = os.path.join(project_dir, 'data\\params.json')


# flair
def flair_ner_cell(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.flair, os.path.join(project_dir, 'mass\\low_level\\ner_flair.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


# deeppavlov
def deeppavlov_ner_cell(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.deeppavlov, os.path.join(project_dir, 'mass\\low_level\\ner_deeppavlov.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


# spacy
def spacy_ner_cell(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.spacy, os.path.join(project_dir, 'mass\\low_level\ner_spacy.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


# nltk stanford
def nltk_stanford_ner_cell(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.nltk, os.path.join(project_dir, 'mass\\low_level\\ner_nltk_stanford.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


