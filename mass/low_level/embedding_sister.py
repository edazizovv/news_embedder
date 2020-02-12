import json
import numpy
import pandas
import sister

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

aggregating_strategy = param['agg']
embedding = None
if aggregating_strategy == 'mean':
    embedding = sister.MeanEmbedding(lang="en")
if embedding is None:
    raise KeyError("Insufficient vespine gas")

result = []
for x in array:
    sentence = x
    result.append(embedding(sentence).reshape(1, -1))

result = numpy.concatenate(result, axis=0)

print('saving')
pandas.DataFrame(result).to_excel('./data/gained.xlsx', index=False)
