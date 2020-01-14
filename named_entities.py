
# from pytorch_pretrained_bert import BertForTokenClassification
# https://github.com/duanzhihua/pytorch-pretrained-BERT
# https://colab.research.google.com/drive/1JxWdw1BjXZCFC2a8IwtZxvvq4rFGcxas#scrollTo=b58Wy9cN9NAA
# https://huggingface.co/transformers/v2.2.0/main_classes/model.html
# https://huggingface.co/transformers/pretrained_models.html

# ======================================================================================================================
"""
Functional interface is the following:
input: 'x' as text and '*args'
output: a dictionary with the following structure: {<named entity token>: [<NER tag>, ...]}

"""
# ======================================================================================================================

# PART 1. COLLECTORS

# PART 2. SEEKERS

# flar:         https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md

from flair.models import SequenceTagger
from flair.data import Sentence

# tested
# standardised
def flair_ner_cell(x, *args):

    tagger = SequenceTagger.load('ner')
    sentence = Sentence(x)
    tagger.predict(sentence)
    ah = sentence.to_dict(tag_type='ner')
    aah = ah['entities']
    enha = {}
    for j in range(len(aah)):
        token = aah[j]['text']
        code = aah[j]['type']
        if token in list(enha.keys()):
            if code not in enha[token]:
                enha[token].append(code)
        else:
            enha[token] = [code]
    return enha


# deeppavlov    https://docs.deeppavlov.ai/en/master/features/models/ner.html
# So, all you need is to run python -m deeppavlov install ... (ex: ner_ontonotes_bert_mult)
# Or install bert_dp manually: pip install -r deeppavlov/requirements/bert_dp.txt

from deeppavlov import configs, build_model
# tested
# standardised
def deeppavlov_ner_cell(x, *args):

    which = args[0]

    ner_model = None
    if which == 'onto_bert_mult':
        ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)  # done
    if which == 'onto_bert':
        ner_model = build_model(configs.ner.ner_ontonotes_bert, download=True)  # done
    if which == 'onto':
        ner_model = build_model(configs.ner.ner_ontonotes, download=True)  # done
    if which == 'conl_bert':
        ner_model = build_model(configs.ner.ner_conll2003_bert, download=True)  # done
    if which == 'conl':
        ner_model = build_model(configs.ner.ner_conll2003, download=True)  # done
    if which == 'dstc2':
        ner_model = build_model(configs.ner.ner_dstc2, download=True)  # done, but miss

    if ner_model is None:
        raise ValueError("Insufficient vespine gas")

    y = ner_model([x])

    enha = {}
    current_token_l = ''
    for j in range(len(y[1][0])):

        token = y[0][0][j]
        code = y[1][0][j]

        if code != 'O':

            code_mark = code[0]
            code_label = code[2:]

            if code_mark == 'B':
                current_token_l = token

            if code_mark == 'I':
                del enha[current_token_l]
                current_token_l = current_token_l + ' ' + token

            if current_token_l in list(enha.keys()):
                if code_label not in enha[current_token_l]:
                    enha[current_token_l].append(code_label)
            else:
                enha[current_token_l] = [code_label]

    return enha

# spacy:        https://www.geeksforgeeks.org/python-named-entity-recognition-ner-using-spacy/
# python -m spacy download en_core_web_sm

import spacy
# tested
# standardised
def spacy_ner_cell(x, *args):

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(x)
    enha = {}
    for x in doc.ents:
        token = x.text
        code = x.label_

        if token in list(enha.keys()):
            if code not in enha[token]:
                enha[token].append(code)
        else:
            enha[token] = [code]

    return enha

# stanford nlp + nltk

# https://textminingonline.com/how-to-use-stanford-named-entity-recognizer-ner-in-python-nltk-and-other-programming-languages

from nltk.tag import StanfordNERTagger
#from nltk.tag.corenlp import CoreNLPNERTagger

import os
java_path = "C:/Program Files/Java/jdk-13.0.1/bin/java.exe"
os.environ['JAVAHOME'] = java_path

a1 = 'C:\\Users\\MainUser\\OneDrive\\RAMP-EXTERNAL\\IP-02\\OSTRTA\\models\\stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.crf.ser.gz'
a2 = 'C:\\Users\\MainUser\\OneDrive\\RAMP-EXTERNAL\\IP-02\\OSTRTA\\models\\stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.prop'

b = 'C:\\Users\\MainUser\\OneDrive\\RAMP-EXTERNAL\\IP-02\\OSTRTA\\models\\stanford-ner-2018-10-16\\stanford-ner.jar'
# tested
# standardised
def nltk_stanford_ner_cell(x, *args):

    st = StanfordNERTagger(a1, b)
    #st = CoreNLPNERTagger(a1, b)
    result = st.tag(x.split())

    enha = {}
    for x in result:
        token = x[0]
        code = x[1]

        if code != 'O':
            if token in list(enha.keys()):
                if code not in enha[token]:
                    enha[token].append(code)
            else:
                enha[token] = [code]

    return enha


# nltk
# https://nlpforhackers.io/named-entity-extraction/

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
# tested
# misunderstood ?
def nltk_ner(x, *args):

    ne_tree = ne_chunk(pos_tag(word_tokenize(x)))
    iob_tagged = tree2conlltags(ne_tree)
    ne_tree = conlltags2tree(iob_tagged)
    return ne_tree




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

