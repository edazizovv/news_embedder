from textblob import TextBlob

import numpy
import pandas

result = []
columns = ['polarity', 'subjectivity']


in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

values = None
for x in array:
    blob = TextBlob(x)
    values = numpy.array([blob.sentiment.polarity, blob.sentiment.subjectivity])
    result.append(values.reshape(1, -1))
result = numpy.concatenate(result, axis=0)

columns = [('TEXTBLOB' + '__' + column) for column in columns]
data = pandas.DataFrame(data=result, columns=columns)
print('saved')
data.to_excel('./data/gained.xlsx', index=False)

