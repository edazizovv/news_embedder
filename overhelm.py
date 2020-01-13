from sentiment import flair_assessor, nltk_assessor, textblob_assessor, pattern_assessor


def over(data_frame, text_column, verbose=True):
    """
    text_column str
    data_frame  pandas.DataFrame
    """

    names = ['flair_positive', 'flair_negative', 'nltk_neg', 'nltk_neu', 'nltk_pos', 'nltk_compound',
             'textblob_polarity', 'textblob_subjectivity', 'pattern_polarity', 'pattern_subjectivity']  # names for data_frame columns we create
    functions = [flair_assessor, flair_assessor, nltk_assessor, nltk_assessor, nltk_assessor, nltk_assessor,
                 textblob_assessor, textblob_assessor, pattern_assessor, pattern_assessor]  # functions we apply to gain each column
    returns = ['positive', 'negative', 'neg', 'neu', 'pos', 'compound',
               'polarity', 'subjectivity', 'polarity', 'subjectivity']  # dict keywords for the returned results from applied functions
    n = len(names)
    for j in range(n):
        data_frame[names[j]] = data_frame[text_column].apply(func=functions[j], args=[returns[j]])
        if verbose:
            print('{} out of {} done'.format((j + 1), n))
    return data_frame



'''
Maskers to recognise NER results from texts. 
They are also used in forming columns names (with adding to them 'model code' + ' | ').
'''
"""
# flair
holy_vector = [x['text'] + ' | ' + x['type'] for x in ah['entities']]
"""
"""
# deeppavlov
n = len(res[0][0])
# we ignore ordering of entities; we care only about general flags
hony_vector = [res[0][0][j] + ' | ' + res[1][0][j][(res[1][0][j].find('-') + 1):] for j in range(n) if res[1][0][j] != 'O']
"""



