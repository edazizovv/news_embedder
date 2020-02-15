import json
import numpy
import pandas
from flair.data import Sentence
from flair.models import TextClassifier

classifier = TextClassifier.load('en-sentiment')

result = []
columns = ['positive', 'negative']

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

in_data = pandas.read_excel(param['data']['opened'])
array = in_data[param['data']['text']].values

for x in array:
    sentence = Sentence(x)

    # predict NER tags
    predicted = classifier.predict(sentence)

    values = None
    if predicted[0].labels[0].value == 'NEGATIVE':
        values = numpy.array([0, predicted[0].labels[0].score])
        # score['positive'] = 0
        # score['negative'] = predicted[0].labels[0].score
    if predicted[0].labels[0].value == 'POSITIVE':
        values = numpy.array([predicted[0].labels[0].score, 0])
        # score['positive'] = predicted[0].labels[0].score
        # score['negative'] = 0
    result.append(values.reshape(1, -1))
result = numpy.concatenate(result, axis=0)
# return score[key]
# return score

# columns = [('FLAIR' + '__' + column) for column in columns]
data = pandas.DataFrame(data=result, columns=columns)
print('saving')
# data.to_excel('./data/gained.xlsx', index=False)

if 'code' in param:
    code_ = param['model']['code'] + '_'
else:
    code_ = ''
columns = {j: 'S_FLR_{}{}'.format(code_, j) for j in data.columns.values}
pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
