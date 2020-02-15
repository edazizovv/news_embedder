import os
import json
import pandas
import subprocess
from configuration import form_factor

current_wd = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(__file__))
params = os.path.join(project_dir, 'data\\params.json')


def flair_embedder(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.flair, os.path.join(project_dir, 'mass\\low_level\\embedding_flair.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


def sister_embedder(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.sister, os.path.join(project_dir, 'mass\\low_level\\embedding_sister.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


def spacy_embedder(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.spacy, os.path.join(project_dir, 'mass\\low_level\\embedding_spacy.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


def use_embedder(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.use, os.path.join(project_dir, 'mass\\low_level\\embedding_use.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data

