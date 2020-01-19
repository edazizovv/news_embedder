import pandas

from configuration import GeneralConfig, SentimentConfig


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



def ner():
    ...

def embedding():
    ...





