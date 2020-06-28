from textblob import TextBlob

import json
import numpy
import pandas

result = []
columns = ['polarity', 'subjectivity']

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

# in_data = pandas.read_excel(param['data']['opened'])
in_data = pandas.read_csv(param['data']['opened'], sep=';')
array = in_data[param['data']['text']].values

values = None
for x in array:
    blob = TextBlob(x)
    values = numpy.array([blob.sentiment.polarity, blob.sentiment.subjectivity])
    result.append(values.reshape(1, -1))
result = numpy.concatenate(result, axis=0)

# columns = [('TEXTBLOB' + '__' + column) for column in columns]
data = pandas.DataFrame(data=result, columns=columns)
print('saving')
# data.to_excel('./data/gained.xlsx', index=False)

if 'code' in param:
    code_ = param['model']['code'] + '_'
else:
    code_ = ''
columns = {j: 'S_TBB_{}{}'.format(code_, j) for j in data.columns.values}
# pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
pandas.DataFrame(data=result, columns=columns).to_csv(param['data']['closed'], index=False, sep=';')
