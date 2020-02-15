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
                                       "I wish he would take a better position one day.",
                                       "Apple and Microsoft sign a new contract for $1.2M",
                                       "A new trade deal has been made, Trump says",
                                       "Elon Musk's invaders caused a global meltdown on Mars, aliens' press-release claims"]})

from overhelm import sentiment_pool, ner_pool, embedding_pool
from configuration import Config_new as Config
config = Config()
config.model = {}

import time
run_time = time.time()
n_char = data['Text'].apply(lambda x: len(x)).sum()
print('Size of the data being treated is:\n\tN of texts = {}\n\tTotal N of characters = {}'.format(data.shape[0], n_char))

# result_data = ner_pool(data, ['spacy'], config)
result_data = sentiment_pool(data, ['textblob'], config)
# result_data = embedding_pool(data, ['use'], config)
run_time = time.time() - run_time
print('Total run time = {0:.2f} seconds'.format(run_time))

