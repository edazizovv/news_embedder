import os
import json
import pandas
import subprocess
from news_embedder.configuration import form_factor

current_wd = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(__file__))
params = os.path.join(project_dir, 'data\\params.json')


def flair_embedder(data, config):
    # data.to_excel(config.data.opened, index=False)
    data.to_csv(config.data.opened, index=False, sep=';')
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.flair, os.path.join(project_dir, 'mass\\low_level\\embedding_flair.py')], capture_output=True)
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


def sister_embedder(data, config):
    # data.to_excel(config.data.opened, index=False)
    data.to_csv(config.data.opened, index=False, sep=';')
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.sister, os.path.join(project_dir, 'mass\\low_level\\embedding_sister.py')], capture_output=True)
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


def spacy_embedder(data, config):
    # data.to_excel(config.data.opened, index=False)
    data.to_csv(config.data.opened, index=False, sep=';')
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.spacy, os.path.join(project_dir, 'mass\\low_level\\embedding_spacy.py')], capture_output=True)
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


def use_embedder(data, config):
    # data.to_excel(config.data.opened, index=False)
    data.to_csv(config.data.opened, index=False, sep=';')
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.use, os.path.join(project_dir, 'mass\\low_level\\embedding_use.py')], capture_output=True)
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

