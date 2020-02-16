from pattern.web import plaintext
from pattern.en import polarity, subjectivity

import json
import pandas
import numpy

result = []
columns = ['polarity', 'subjectivity']

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

in_data = pandas.read_excel(param['data']['opened'])
array = in_data[param['data']['text']].values

values = None
for x in array:
    pt = plaintext(x)
    values = numpy.array([polarity(pt), subjectivity(pt)])
    result.append(values.reshape(1, -1))
result = numpy.concatenate(result, axis=0)

# columns = [('PATTERN' + '__' + column) for column in columns]
data = pandas.DataFrame(data=result, columns=columns)
print('saving')
# data.to_excel('./data/gained.xlsx', index=False)

if 'code' in param:
    code_ = param['model']['code'] + '_'
else:
    code_ = ''
columns = {j: 'S_PTT_{}{}'.format(code_, j) for j in data.columns.values}
pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
