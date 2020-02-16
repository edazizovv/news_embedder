import os
import json

project_dir = os.path.dirname(__file__)


def form_factor(config):
    adds = {'jdk': config.adds.jdk,
            'stanford_ner': config.adds.stanford_ner,
            'gz': config.adds.gz}
    data = {'text': config.data.text,
            'opened': config.data.opened,
            'closed': config.data.closed}
    model = config.model
    form = {'adds': adds,
            'data': data,
            'model': model}
    return form


class VirtualConfig:
    def __init__(self):
        with open(os.path.join(project_dir, 'settings\\virtual.json'), 'r') as js:
            param = json.load(js)
        self.flair = param['flair']
        self.nltk = param['nltk']
        self.textblob = param['textblob']
        self.pattern = param['pattern']
        self.deeppavlov = param['deeppavlov']
        self.spacy = param['spacy']
        self.sister = param['sister']
        self.use = param['use']


class AddsConfig:
    def __init__(self):
        with open(os.path.join(project_dir, 'settings\\adds.json'), 'r') as js:
            param = json.load(js)
        self.jdk = param['jdk']
        self.stanford_ner = param['stanford_ner']
        self.gz = param['gz']


class DataConfig:
    def __init__(self, text):
        with open(os.path.join(project_dir, 'settings\\data.json'), 'r') as js:
            param = json.load(js)
        self.text = text
        self.opened = os.path.join(project_dir, 'data\\source.xlsx')
        self.closed = os.path.join(project_dir, 'data\\gained.xlsx')


class Config:
    def __init__(self, text='Text'):
        self.virtual = VirtualConfig()
        self.adds = AddsConfig()
        self.data = DataConfig(text)
        self.model = None

