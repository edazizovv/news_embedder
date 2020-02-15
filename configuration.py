import json


def form_factor(config):
    adds = {'jdk': config.adds.jdk,
            'stanford_ner': config.adds.stanford_ner,
            'gz': config.adds.gz}
    data = {'text': config.data.text}
    model = config.model
    form = {'adds': adds,
            'data': data,
            'model': model}
    return form


class VirtualConfig_new:
    def __init__(self):
        with open('./settings/virtual.json', 'r') as js:
            param = json.load(js)
        self.flair = param['flair']
        self.nltk = param['nltk']
        self.textblob = param['textblob']
        self.pattern = param['pattern']
        self.deeppavlov = param['deeppavlov']
        self.spacy = param['spacy']
        self.sister = param['sister']
        self.use = param['use']


class AddsConfig_new:
    def __init__(self):
        with open('./settings/adds.json', 'r') as js:
            param = json.load(js)
        self.jdk = param['jdk']
        self.stanford_ner = param['stanford_ner']
        self.gz = param['gz']


class DataConfig_new:
    def __init__(self, text):
        with open('./settings/data.json', 'r') as js:
            param = json.load(js)
        self.text = text


class Config_new:
    def __init__(self, text='Text'):
        self.virtual = VirtualConfig_new()
        self.adds = AddsConfig_new()
        self.data = DataConfig_new(text)
        self.model = None





class Config:
    def __init__(self):
        self.virtual = VirtualConfig()
        self.paths = Paths()
        self.text_column = 'Text'
        self.date_column = 'DateTime'
        self.params = None


class VirtualConfig:
    def __init__(self):
        self.flair = 'E:/venv/hazard_flair/python.exe'
        self.nltk = 'E:/venv/hazard_nltk/python.exe'
        self.textblob = 'E:/venv/hazard_textblob/python.exe'
        self.pattern = 'E:/venv/hazard_pattern/python.exe'
        self.deeppavlov = 'E:/venv/hazard_deeppavlov/python.exe'
        self.spacy = 'E:/venv/hazard_spacy/python.exe'
        self.sister = 'E:/venv/hazard_sister/python.exe'
        self.use = 'E:/venv/hazard_use/python.exe'


class Paths:
    def __init__(self):
        self.store = './data/'
        self.opened = 'source.xlsx'
        self.closed = 'gained.xlsx'

