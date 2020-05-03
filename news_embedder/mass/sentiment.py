import os
import json
import pandas
import subprocess
from news_embedder.configuration import form_factor

current_wd = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(__file__))
params = os.path.join(project_dir, 'data\\params.json')


def flair_assessor(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.flair, os.path.join(project_dir, 'mass\\low_level\\sentiment_flair.py')], capture_output=True)
    if result.returncode == 0:
        pass
    else:
        print(result.stderr)
        raise Exception
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


def nltk_assessor(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.nltk, os.path.join(project_dir, 'mass\\low_level\\sentiment_nltk.py')], capture_output=True)
    if result.returncode == 0:
        pass
    else:
        print(result.stderr)
        raise Exception
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


def textblob_assessor(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.textblob, os.path.join(project_dir, 'mass\\low_level\\sentiment_textblob.py')], capture_output=True)
    if result.returncode == 0:
        pass
    else:
        print(result.stderr)
        raise Exception
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


def pattern_assessor(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    result = subprocess.run([config.virtual.pattern, os.path.join(project_dir, 'mass\\low_level\\sentiment_pattern.py')], capture_output=True)
    if result.returncode == 0:
        pass
    else:
        print(result.stderr)
        raise Exception
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data
