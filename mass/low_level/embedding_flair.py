import json
import numpy
import pandas
from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.embeddings import DocumentPoolEmbeddings, DocumentRNNEmbeddings, Sentence

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

layers = param['layers']
layers_source = param['sources']

for j in range(len(layers)):
    if layers_source[j] == 'word':
        layers[j] = WordEmbeddings(layers[j])
    if layers_source[j] == 'flair':
        layers[j] = FlairEmbeddings(layers[j])
    if layers_source[j] != 'word' and layers_source[j] != 'flair':
        raise KeyError("Not Enough Minerals")

embedding = None
if param['agg'] == 'pooling':
    # TODO: check if kwargs work
    embedding = DocumentPoolEmbeddings(layers, **param['options'])
if param['agg'] == 'rnn':
    # TODO: check if kwargs work
    embedding = DocumentRNNEmbeddings(layers, **param['options'])
if embedding is None:
    raise KeyError("Insufficient vespine gas")

result = []
for x in array:
    sentence = Sentence(x)
    embedding.embed(sentence)
    result.append(sentence.embedding.detach().numpy().reshape(1, -1))

result = numpy.concatenate(result, axis=0)

print('saving')
# pandas.DataFrame(result).to_excel('./data/gained.xlsx', index=False)

if 'code' in param:
    code_ = param['code'] + '_'
else:
    code_ = ''
columns = ['E_FLR_{}{}'.format(code_, j) for j in range(result.shape[1])]
pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
