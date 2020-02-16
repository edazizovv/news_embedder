import pandas

from news_embedder.mass.sentiment import flair_assessor, pattern_assessor, textblob_assessor, nltk_assessor
from news_embedder.mass.ner import flair_ner_cell, deeppavlov_ner_cell, spacy_ner_cell, nltk_stanford_ner_cell
from news_embedder.mass.embedding import flair_embedder, sister_embedder, spacy_embedder, use_embedder


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


def ner_pool(data_frame, names, config):
    if 'flair' in names:
        flair_data = flair_ner_cell(data_frame, config)
    else:
        flair_data = pandas.DataFrame()
    if 'deeppavlov' in names:
        deeppavlov_data = deeppavlov_ner_cell(data_frame, config)
    else:
        deeppavlov_data = pandas.DataFrame()
    if 'spacy' in names:
        spacy_data = spacy_ner_cell(data_frame, config)
    else:
        spacy_data = pandas.DataFrame()
    if 'nltk_stanford' in names:
        nltk_stanford_data = nltk_stanford_ner_cell(data_frame, config)
    else:
        nltk_stanford_data = pandas.DataFrame()
    data_frame = pandas.concat(objs=(data_frame, flair_data, deeppavlov_data, spacy_data, nltk_stanford_data), axis=1)
    return data_frame


def embedding_pool(data_frame, names, config):
    if 'flair' in names:
        flair_data = flair_embedder(data_frame, config)
    else:
        flair_data = pandas.DataFrame()
    if 'sister' in names:
        sister_data = sister_embedder(data_frame, config)
    else:
        sister_data = pandas.DataFrame()
    if 'spacy' in names:
        spacy_data = spacy_embedder(data_frame, config)
    else:
        spacy_data = pandas.DataFrame()
    if 'use' in names:
        use_data = use_embedder(data_frame, config)
    else:
        use_data = pandas.DataFrame()
    data_frame = pandas.concat(objs=(data_frame, flair_data, sister_data, spacy_data, use_data), axis=1)
    return data_frame





