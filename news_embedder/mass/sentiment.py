import os
import json
import pandas
import subprocess
from news_embedder.configuration import form_factor

current_wd = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(__file__))
params = os.path.join(project_dir, 'data\\params.json')


# flair
def flair_assessor(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.flair, os.path.join(project_dir, 'mass\\low_level\\sentiment_flair.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data


# nltk
def nltk_assessor(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.nltk, os.path.join(project_dir, 'mass\\low_level\\sentiment_nltk.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data

# textblob
def textblob_assessor(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.textblob, os.path.join(project_dir, 'mass\\low_level\\sentiment_textblob.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data

# pattern
def pattern_assessor(data, config):
    data.to_excel(config.data.opened, index=False)
    form = form_factor(config)
    with open(params, 'w') as js:
        json.dump(form, js)
    os.chdir(project_dir)
    subprocess.call([config.virtual.pattern, os.path.join(project_dir, 'mass\\low_level\\sentiment_pattern.py')])
    os.chdir(current_wd)
    data = pandas.read_excel(config.data.closed)
    return data
