
# https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md

# from pytorch_pretrained_bert import BertForTokenClassification
# https://github.com/duanzhihua/pytorch-pretrained-BERT
# https://colab.research.google.com/drive/1JxWdw1BjXZCFC2a8IwtZxvvq4rFGcxas#scrollTo=b58Wy9cN9NAA
# https://huggingface.co/transformers/v2.2.0/main_classes/model.html
# https://huggingface.co/transformers/pretrained_models.html
# https://docs.deeppavlov.ai/en/master/features/models/ner.html

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
