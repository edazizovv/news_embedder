import json
import pandas
import subprocess
from configuration import form_factor

opened = './data/source.xlsx'
closed = './data/gained.xlsx'


def flair_embedder(data, config):
    data.to_excel(opened, index=False)
    form = form_factor(config)
    with open('./data/params.json', 'w') as js:
        json.dump(form, js)
    subprocess.call([config.virtual.flair, './mass/low_level/embedding_flair.py'])
    data = pandas.read_excel(closed)
    return data


def sister_embedder(data, config):
    data.to_excel(opened, index=False)
    form = form_factor(config)
    with open('./data/params.json', 'w') as js:
        json.dump(form, js)
    subprocess.call([config.virtual.sister, './mass/low_level/embedding_sister.py'])
    data = pandas.read_excel(closed)
    return data


def spacy_embedder(data, config):
    data.to_excel(opened, index=False)
    form = form_factor(config)
    with open('./data/params.json', 'w') as js:
        json.dump(form, js)
    subprocess.call([config.virtual.spacy, './mass/low_level/embedding_spacy.py'])
    data = pandas.read_excel(closed)
    return data


def use_embedder(data, config):
    data.to_excel(opened, index=False)
    form = form_factor(config)
    with open('./data/params.json', 'w') as js:
        json.dump(form, js)
    subprocess.call([config.virtual.use, './mass/low_level/embedding_use.py'])
    data = pandas.read_excel(closed)
    return data

