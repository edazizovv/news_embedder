import os
import json
import pandas
import subprocess
from news_embedder.configuration import form_factor

current_wd = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(__file__))
params = os.path.join(project_dir, 'data\\params.json')


def flair_ner_cell(data, config):
    # data.to_excel(config.data.opened, index=False)
    data.to_csv(config.data.opened, index=False, sep=';')
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.flair, os.path.join(project_dir, 'mass\\low_level\\ner_flair.py')], capture_output=True)
    if result.returncode == 0:
        pass
    else:
        print('\n\n')
        print(result.stderr)
        print('\n\n')
        raise Exception
    os.chdir(current_wd)
    # data = pandas.read_excel(config.data.closed)
    data = pandas.read_csv(config.data.closed, sep=';')
    return data


def deeppavlov_ner_cell(data, config):
    # data.to_excel(config.data.opened, index=False)
    data.to_csv(config.data.opened, index=False, sep=';')
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.deeppavlov, os.path.join(project_dir, 'mass\\low_level\\ner_deeppavlov.py')], capture_output=True)
    if result.returncode == 0:
        pass
    else:
        print('\n\n')
        print(result.stderr)
        print('\n\n')
        raise Exception
    os.chdir(current_wd)
    # data = pandas.read_excel(config.data.closed)
    data = pandas.read_csv(config.data.closed, sep=';')
    return data


def spacy_ner_cell(data, config):
    # data.to_excel(config.data.opened, index=False)
    data.to_csv(config.data.opened, index=False, sep=';')
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.spacy, os.path.join(project_dir, 'mass\\low_level\\ner_spacy.py')], capture_output=True)
    if result.returncode == 0:
        pass
    else:
        print('\n\n')
        print(result.stderr)
        print('\n\n')
        raise Exception
    os.chdir(current_wd)
    # data = pandas.read_excel(config.data.closed)
    data = pandas.read_csv(config.data.closed, sep=';')
    return data


def nltk_stanford_ner_cell(data, config):
    # data.to_excel(config.data.opened, index=False)
    data.to_csv(config.data.opened, index=False, sep=';')
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.nltk, os.path.join(project_dir, 'mass\\low_level\\ner_nltk_stanford.py')], capture_output=True)
    if result.returncode == 0:
        pass
    else:
        print('\n\n')
        print(result.stderr)
        print('\n\n')
        raise Exception
    os.chdir(current_wd)
    # data = pandas.read_excel(config.data.closed)
    data = pandas.read_csv(config.data.closed, sep=';')
    return data


