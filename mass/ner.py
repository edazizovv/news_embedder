import pandas
import subprocess


# flair
def flair_ner_cell(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    subprocess.call([config.virtual.flair, './mass/low_level/ner_flair.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


# deeppavlov
def deeppavlov_ner_cell(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    if config.param == 'onto_bert_mult':
        subprocess.call([config.virtual.deeppavlov, './mass/low_level/ner_deeppavlov_onto_bert_mult.py'])
    if config.param == 'onto_bert':
        subprocess.call([config.virtual.deeppavlov, './mass/low_level/ner_deeppavlov_onto_bert.py'])
    if config.param == 'onto':
        subprocess.call([config.virtual.deeppavlov, './mass/low_level/ner_deeppavlov_onto.py'])
    if config.param == 'conl_bert':
        subprocess.call([config.virtual.deeppavlov, './mass/low_level/ner_deeppavlov_conl_bert.py'])
    if config.param == 'conl':
        subprocess.call([config.virtual.deeppavlov, './mass/low_level/ner_deeppavlov_conl.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


# spacy
def spacy_ner_cell(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    if config.param == 'sm':
        subprocess.call([config.virtual.spacy, './mass/low_level/ner_spacy_sm.py'])
    if config.param == 'md':
        subprocess.call([config.virtual.spacy, './mass/low_level/ner_spacy_md.py'])
    if config.param == 'lg':
        subprocess.call([config.virtual.spacy, './mass/low_level/ner_spacy_lg.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data

# nltk stanford
def nltk_stanford_ner_cell(data, config):
    data.to_excel((config.paths.store + config.paths.opened), index=False)
    subprocess.call([config.virtual.nltk, './mass/low_level/ner_nltk_stanford.py'])
    data = pandas.read_excel((config.paths.store + config.paths.closed))
    return data


