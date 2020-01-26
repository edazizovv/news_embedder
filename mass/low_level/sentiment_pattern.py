print('hello there?')
from pattern.web import plaintext
from pattern.en import polarity, subjectivity

import pandas
import numpy

result = []
columns = ['polarity', 'subjectivity']

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

values = None
for x in array:
    pt = plaintext(x)
    values = numpy.array([polarity(pt), subjectivity(pt)])
    result.append(values.reshape(1, -1))
result = numpy.concatenate(result, axis=0)

columns = [('PATTERN' + '__' + column) for column in columns]
data = pandas.DataFrame(data=result, columns=columns)
print('saved')
data.to_excel('./data/gained.xlsx', index=False)
