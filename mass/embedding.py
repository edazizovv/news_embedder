import json
import pandas
import subprocess


def flair_embedder(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    with open('./data/params.json', 'w') as param:
        json.dump(config.params, param)
    subprocess.call([config.virtual.flair, './mass/low_level/embedding_flair.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


def sister_embedder(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    with open('./data/params.json', 'w') as param:
        json.dump(config.params, param)
    subprocess.call([config.virtual.sister, './mass/low_level/embedding_sister.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


def spacy_embedder(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    with open('./data/params.json', 'w') as param:
        json.dump(config.params, param)
    subprocess.call([config.virtual.spacy, './mass/low_level/embedding_spacy.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


def use_embedder(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    with open('./data/params.json', 'w') as param:
        json.dump(config.params, param)
    subprocess.call([config.virtual.use, './mass/low_level/embedding_use.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data

