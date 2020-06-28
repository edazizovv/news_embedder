import json
import numpy
import pandas
import tensorflow_hub

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

# in_data = pandas.read_excel(param['data']['opened'])
in_data = pandas.read_csv(param['data']['opened'], sep=';')
array = in_data[param['data']['text']].values

embed = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


result = []
for x in array:
    resu = embed([x])
    result.append(resu.numpy())

result = numpy.concatenate(result, axis=0)

print('saving')
# pandas.DataFrame(result).to_excel('./data/gained.xlsx', index=False)

if 'code' in param:
    code_ = param['model']['code'] + '_'
else:
    code_ = ''
columns = ['E_USE_{}{}'.format(code_, j) for j in range(result.shape[1])]
# pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
pandas.DataFrame(data=result, columns=columns).to_csv(param['data']['closed'], index=False, sep=';')
