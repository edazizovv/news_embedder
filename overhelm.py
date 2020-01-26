import pandas

from mass.sentiment import flair_assessor, pattern_assessor, textblob_assessor, nltk_assessor

"""
def sentiment_pool(data_frame):
    general_configurator = GeneralConfig()
    sentiment_configurator = SentimentConfig()
    data = data_frame[general_configurator.text].values
    for name in sentiment_configurator.names:
        func = sentiment_configurator.functions[name]
        args = sentiment_configurator.parameters[name]
        columns, values = func(data, *args)
        columns = [(name + '__' + column) for column in columns]
        add_data = pandas.DataFrame(data=values, columns=columns)
        data_frame = pandas.concat((data_frame, add_data), axis=1)
    return data_frame
"""
def sentiment_pool(data_frame, names, config):
    if 'flair' in names:
        flair_data = flair_assessor(data_frame, config)
    else:
        flair_data = pandas.DataFrame()
    if 'pattern' in names:
        pattern_data = pattern_assessor(data_frame, config)
    else:
        pattern_data = pandas.DataFrame()
    if 'textblob' in names:
        textblob_data = textblob_assessor(data_frame, config)
    else:
        textblob_data = pandas.DataFrame()
    if 'nltk' in names:
        nltk_data = nltk_assessor(data_frame, config)
    else:
        nltk_data = pandas.DataFrame()

    data_frame = pandas.concat(objs=(data_frame, flair_data, pattern_data, textblob_data, nltk_data), axis=1)
    return data_frame

def ner():
    ...

def embedding():
    ...





