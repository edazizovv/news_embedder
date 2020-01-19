# TODO: neither with pip, nor with conda file requirements.txt does not work, fix it!

"""
from sentiment import flair_assessor, nltk_assessor, textblob_assessor, pattern_assessor, stanford_assessor


# sentence = "Quick red fox jumps over a lazy dog. Wow really?"  #
sentence = "Quick red fox jumps over a lazy dog. So cute!"  #
# sentence = "Quick red fox jumps over a lazy dog. I hate them!"  #
# sentence = "My boss is an idiot. I wish he would be fired one day."  #
# sentence = "My boss is amazing! I wish he would take a better position one day."  #

# sentence = "Quick red fox jumps over a lazy dog"  #
# sentence = "Wow really?"  #
# sentence = "So cute!"  #
# sentence = "I hate them!"  #
# sentence = "My boss is an idiot. "  #
# sentence = "I wish he would be fired one day."  #
# sentence = "My boss is amazing! "  #
# sentence = "I wish he would take a better position one day."  #

args = ['polarity']
# result = flair_assessor(sentence, *args)
# result = nltk_assessor(sentence, *args)
# result = textblob_assessor(sentence, *args)
result = pattern_assessor(sentence, *args)
"""
"""
import pandas
from overhelm import over

vex = ["Quick red fox jumps over a lazy dog. Wow really?",
       "Quick red fox jumps over a lazy dog. So cute!",
       "Quick red fox jumps over a lazy dog. I hate them!",
       "My boss is an idiot. I wish he would be fired one day.",
       "My boss is amazing! I wish he would take a better position one day.",
       "Quick red fox jumps over a lazy dog",
       "Wow really?",
       "So cute!",
       "I hate them!",
       "My boss is an idiot. ",
       "I wish he would be fired one day.",
       "My boss is amazing! ",
       "I wish he would take a better position one day."]
data = {'ix': list(range(len(vex))),
        'texts': vex}
data_frame = pandas.DataFrame(data=data)

data_frame = over(data_frame=data_frame, text_column='texts')
"""

'''
from solo.sentiment import nltk_assessor as assessor

x = "Quick red fox jumps over a lazy dog."
args = ['a']
y = assessor(x, *args)
'''
'''
from named_entities import nltk_stanford_ner_cell as cell

x = "Quick red fox ate an apple. Morgan Chase is bankrupt"
args = ['conl']
y = cell(x, *args)
'''
'''
from embedding import spacy_embeddings as embe
# from embedding import flair_word_embeddings as worders
x = "Quick red fox ate an apple. Morgan Chase is bankrupt"

# layers = worders(['glove', 'news-forward', 'news-backward'])
# args = [layers,
#         'rnn',
#         {'rnn_type': 'GRU'}]
args = ['sm']
y = embe(x, *args)
'''


from overhelm import sentiment_pool
import pandas
data = pandas.DataFrame(data={'Text': ["Quick red fox jumps over a lazy dog. Wow really?",
                                       "Quick red fox jumps over a lazy dog. So cute!",
                                       "Quick red fox jumps over a lazy dog. I hate them!",
                                       "My boss is an idiot. I wish he would be fired one day.",
                                       "My boss is amazing! I wish he would take a better position one day.",
                                       "Quick red fox jumps over a lazy dog",
                                       "Wow really?",
                                       "So cute!",
                                       "I hate them!",
                                       "My boss is an idiot. ",
                                       "I wish he would be fired one day.",
                                       "My boss is amazing! ",
                                       "I wish he would take a better position one day."]})

result_data = sentiment_pool(data)
