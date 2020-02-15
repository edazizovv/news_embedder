import json
import numpy
import pandas

from flair.models import SequenceTagger
from flair.data import Sentence

with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

in_data = pandas.read_excel(param['data']['opened'])
array = in_data[param['data']['text']].values

tagger = SequenceTagger.load('ner')

# Step 2. Make our data (with the vocabulary navigating columns)

start = True
start_len = 0
j = 0
result = []
columns = []
for y in array:
    sentence = Sentence(y)
    tagger.predict(sentence)
    ah = sentence.to_dict(tag_type='ner')
    aah = ah['entities']
    enha = {}
    for j in range(len(aah)):
        token = aah[j]['text']
        code = aah[j]['type']
        if token in list(enha.keys()):
            if code not in enha[token]:
                enha[token].append(code)
        else:
            enha[token] = [code]

    # h = list(enha.keys())
    add_c = []
    # keys_s, values_s = list(enha.keys()), list(enha.values())
    #print('================')
    #print(y)
    for kk in enha.keys():
        for vv in enha[kk]:
            appie = "[{}]_['{}']".format(kk, vv)
            add_c.append(appie)
    outers = [z for z in add_c if z not in columns]
    columns = columns + outers
    h = outers
    #print('-------------')
    #print(enha)
    #print(h)
    #print(columns)
    if len(h) > 0:
        start_len = start_len + len(h)
        if start:
            start = False
        else:
            for g in range(len(result)):
                gle = len(h)
                result[g] = numpy.concatenate((result[g], numpy.zeros(shape=(1, gle))), axis=1)

        values = numpy.zeros(shape=(1, start_len))

        for key in enha.keys():
            for value in enha[key]:
                appi = "[{}]_['{}']".format(key, value)
                ix = columns.index(appi)
                values[0, ix] = 1
        result.append(values)

        #print(result)
    else:
        result.append(numpy.zeros(shape=(1, start_len)))

result = numpy.concatenate(result, axis=0)

data = pandas.DataFrame(data=result, columns=columns)
print('saving')
# data.to_excel('./data/gained.xlsx', index=False)

if 'code' in param:
    code_ = param['model']['code'] + '_'
else:
    code_ = ''
columns = {j: 'R_FLR_{}{}'.format(code_, j) for j in data.columns.values}
pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
