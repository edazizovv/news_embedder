
# flar:         https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md
# deeppavlov    https://docs.deeppavlov.ai/en/master/features/models/ner.html
# spacy:        https://www.geeksforgeeks.org/python-named-entity-recognition-ner-using-spacy/

# from pytorch_pretrained_bert import BertForTokenClassification
# https://github.com/duanzhihua/pytorch-pretrained-BERT
# https://colab.research.google.com/drive/1JxWdw1BjXZCFC2a8IwtZxvvq4rFGcxas#scrollTo=b58Wy9cN9NAA
# https://huggingface.co/transformers/v2.2.0/main_classes/model.html
# https://huggingface.co/transformers/pretrained_models.html


# flair

"""
from flair.models import SequenceTagger
from flair.data import Sentence

tagger = SequenceTagger.load('ner')
#sentence = Sentence('George Washington went to Washington .')
#sentence = Sentence("Intel is loosing it's market .")
sentence = Sentence("Nvidia is loosing it's market .")

tagger.predict(sentence)
#print(sentence.to_tagged_string())
#print(sentence.to_dict(tag_type='ner'))

ah = sentence.to_dict(tag_type='ner')
"""

# deeppavlov

"""
# So, all you need is to run python -m deeppavlov install ... (ex: ner_ontonotes_bert_mult)
# Or install bert_dp manually: pip install -r deeppavlov/requirements/bert_dp.txt

from deeppavlov import configs, build_model

#ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)  # done
#ner_model = build_model(configs.ner.ner_ontonotes_bert, download=True)  # done
#ner_model = build_model(configs.ner.ner_ontonotes, download=True)  # done
#ner_model = build_model(configs.ner.ner_conll2003_bert, download=True)  # done
#ner_model = build_model(configs.ner.ner_conll2003, download=True)  # done
ner_model = build_model(configs.ner.ner_dstc2, download=True)  # done, but miss

res = ner_model(["Intel is loosing it's market ."])
#res = ner_model(["Nvidia is loosing it's market ."])
"""

"""
# spaCy
# python -m spacy download en_core_web_sm

import spacy

nlp = spacy.load('en_core_web_sm')

sentence = "Apple is looking at buying U.K. startup for $1 billion"

doc = nlp(sentence)

for ent in doc.ents:
    print(ent.text, ent.label_)
"""

# stanford nlp + pycorenlp

# !
# java -mx2g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
# !

"""
import pandas
from pycorenlp import StanfordCoreNLP


nlp = StanfordCoreNLP('http://localhost:9000')

def stanford_assessor(frame, column_name):

    # here it analyses every sentence -- but we need to analyse full text
    # need some aggregation?

    sentences = frame[column_name].values.tolist()

    passed = []
    # result["sentences"][0]["entitymentions"]
    for j in range(len(sentences)):
        result = nlp.annotate(sentences[j],
                              properties={
                                  'annotators': 'ner',
                                  'outputFormat': 'json',
                                  'timeout': 10000,
                              })
        s = result["sentences"][j]
        sss = {'index': [s["index"]], 'sentence': [" ".join([t["word"] for t in s["ner"]])]}

        passed.append(pandas.DataFrame(data=sss))

    passed = pandas.concat(objs=passed, axis='index', ignore_index=True)

    return passed
"""


# stanford nlp + nltk
"""
# https://textminingonline.com/how-to-use-stanford-named-entity-recognizer-ner-in-python-nltk-and-other-programming-languages

from nltk.tag import StanfordNERTagger
#from nltk.tag.corenlp import CoreNLPNERTagger

import os
java_path = "C:/Program Files/Java/jdk-13.0.1/bin/java.exe"
os.environ['JAVAHOME'] = java_path

a1 = 'C:\\Sygm\\RAMP\\IP-02\\OSTRTA\\venv\\news_historica\\share\\stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.crf.ser.gz'
a2 = 'C:\\Sygm\\RAMP\\IP-02\\OSTRTA\\venv\\news_historica\\share\\stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.prop'

b = 'C:\\Sygm\\RAMP\\IP-02\\OSTRTA\\venv\\news_historica\\share\\stanford-ner-2018-10-16\\stanford-ner.jar'

st = StanfordNERTagger(a1, b)
#st = CoreNLPNERTagger(a1, b)

reuslt = st.tag('Rami Eid is studying at Stony Brook University in NY'.split())
"""

"""
# nltk 1
# https://nlpforhackers.io/named-entity-extraction/

from nltk import word_tokenize, pos_tag, ne_chunk

sentence = "Mark and John are working at Google."

result = word_tokenize(sentence)
"""

"""
# nltk 2
# https://nlpforhackers.io/named-entity-extraction/

from nltk.chunk import conlltags2tree, tree2conlltags

sentence = "Mark and John are working at Google."
ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))

iob_tagged = tree2conlltags(ne_tree)
#print(iob_tagged)

ne_tree = conlltags2tree(iob_tagged)
print(ne_tree)
"""