

class Config:
    def __init__(self):
        self.virtual = VirtualConfig()
        self.paths = Paths()
        self.text_column = 'Text'
        self.date_column = 'DateTime'


class VirtualConfig:
    def __init__(self):
        self.flair = 'E:/venv/hazard_flair/python.exe'
        self.nltk = 'E:/venv/hazard_nltk/python.exe'
        self.textblob = 'E:/venv/hazard_textblob/python.exe'
        self.pattern = 'E:/venv/hazard_pattern/python.exe'


class Paths:
    def __init__(self):
        self.store = './data/'
        self.opened = 'source.xlsx'
        self.closed = 'gained.xlsx'

