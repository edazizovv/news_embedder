from nltk.sentiment.vader import SentimentIntensityAnalyzer

import json
import numpy
import pandas

result = []
columns = ['neg', 'neu', 'pos', 'compound']

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

# in_data = pandas.read_excel(param['data']['opened'])
in_data = pandas.read_csv(param['data']['opened'], sep=';')
array = in_data[param['data']['text']].values

sid = SentimentIntensityAnalyzer()

values = None
for x in array:
    ss = sid.polarity_scores(x)
    values = numpy.array([ss['neg'], ss['neu'], ss['pos'], ss['compound']])
    result.append(values.reshape(1, -1))
result = numpy.concatenate(result, axis=0)

# columns = [('NLTK' + '__' + column) for column in columns]
data = pandas.DataFrame(data=result, columns=columns)
print('saving')
# data.to_excel('./data/gained.xlsx', index=False)

if 'code' in param:
    code_ = param['model']['code'] + '_'
else:
    code_ = ''
columns = {j: 'S_NTK_{}{}'.format(code_, j) for j in data.columns.values}
# pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
pandas.DataFrame(data=result, columns=columns).to_csv(param['data']['closed'], index=False, sep=';')

