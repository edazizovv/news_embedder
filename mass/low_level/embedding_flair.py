import json
import numpy
import pandas
from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.embeddings import DocumentPoolEmbeddings, DocumentRNNEmbeddings, Sentence

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data[param['data']['text']].values

layers = param['model']['models']
#layers_source = param['model']['sources']

word_models = ['glove', 'turian', 'extvec', 'crawl', 'news', 'twitter', 'en-wiki', 'en-crawl']
flair_models = ['en-forward', 'en-backward', 'news-forward', 'news-backward', 'mix-forward', 'mix-backward']

for j in range(len(layers)):
    if (layers[j] not in word_models) and (layers[j] not in flair_models):
        raise KeyError("Not Enough Minerals")
    if layers[j] in word_models:
        layers[j] = WordEmbeddings(layers[j])
    if layers[j] in flair_models:
        layers[j] = FlairEmbeddings(layers[j])


embedding = None
if param['model']['agg'] == 'pooling':
    # TODO: check if kwargs work
    embedding = DocumentPoolEmbeddings(layers, **param['model']['options'])
if param['model']['agg'] == 'rnn':
    # TODO: check if kwargs work
    embedding = DocumentRNNEmbeddings(layers, **param['model']['options'])
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
    code_ = param['model']['code'] + '_'
else:
    code_ = ''
columns = ['E_FLR_{}{}'.format(code_, j) for j in range(result.shape[1])]
pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
