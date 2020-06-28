import json
import numpy
import pandas
import sister

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

# in_data = pandas.read_excel(param['data']['opened'])
in_data = pandas.read_csv(param['data']['opened'], sep=';')
array = in_data[param['data']['text']].values

aggregating_strategy = param['model']['agg']
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
# pandas.DataFrame(result).to_excel('./data/gained.xlsx', index=False)

if 'code' in param:
    code_ = param['model']['code'] + '_'
else:
    code_ = ''
columns = ['E_SSR_{}{}'.format(code_, j) for j in range(result.shape[1])]
# pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
pandas.DataFrame(data=result, columns=columns).to_csv(param['data']['closed'], index=False, sep=';')
