

class VirtualConfig:
    ...


class OverhelmConfig:
    ...


class SentimentConfig:
    def __init__(self):
        from mass.sentiment import pattern_assessor
        self.names = ['pattern']
        self.functions = {'pattern': pattern_assessor}
        self.parameters = {'pattern': []}

class GeneralConfig:
    def __init__(self):
        self.text = 'Text'
        self.datetime = 'DateTime'

