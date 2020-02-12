import json
import numpy
import pandas
import tensorflow_hub

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

embed = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


result = []
for x in array:
    resu = embed([x])
    result.append(resu.numpy())

result = numpy.concatenate(result, axis=0)

print('saving')
pandas.DataFrame(result).to_excel('./data/gained.xlsx', index=False)
