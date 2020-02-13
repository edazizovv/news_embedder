from textblob import TextBlob

import json
import numpy
import pandas

result = []
columns = ['polarity', 'subjectivity']


in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

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
    code_ = param['code'] + '_'
else:
    code_ = ''
columns = {j: 'S_TBB_{}{}'.format(code_, j) for j in data.columns.values}
data.rename(columns=columns).to_excel('./data/gained.xlsx', index=False)
