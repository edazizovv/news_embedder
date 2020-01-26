from nltk.sentiment.vader import SentimentIntensityAnalyzer

import numpy
import pandas

result = []
columns = ['neg', 'neu', 'pos', 'compound']

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

sid = SentimentIntensityAnalyzer()

values = None
for x in array:
    ss = sid.polarity_scores(x)
    values = numpy.array([ss['neg'], ss['neu'], ss['pos'], ss['compound']])
    result.append(values.reshape(1, -1))
result = numpy.concatenate(result, axis=0)

columns = [('NLTK' + '__' + column) for column in columns]
data = pandas.DataFrame(data=result, columns=columns)
print('saved')
data.to_excel('./data/gained.xlsx', index=False)

