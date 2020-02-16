import json
import numpy
import pandas
import spacy

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

in_data = pandas.read_excel(param['data']['opened'])
array = in_data[param['data']['text']].values

model = param['model']['model']
embedder = None
if model == 'sm':
    embedder = spacy.load('en_core_web_sm')
if model == 'md':
    embedder = spacy.load('en_core_web_md')
if model == 'lg':
    embedder = spacy.load('en_core_web_lg')
if embedder is None:
    raise KeyError("Insufficient vespine gas")


result = []
for x in array:
    doc = embedder(x)
    result.append(doc.vector.reshape(1, -1))

result = numpy.concatenate(result, axis=0)

print('saving')
# pandas.DataFrame(result).to_excel('./data/gained.xlsx', index=False)

if 'code' in param:
    code_ = param['model']['code'] + '_'
else:
    code_ = ''
columns = ['E_SPC_{}{}'.format(code_, j) for j in range(result.shape[1])]
pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
