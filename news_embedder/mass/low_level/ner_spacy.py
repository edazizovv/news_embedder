import json
import numpy
import pandas
import spacy


with open('./data/params.json', 'r') as param_:
    param = json.load(param_)

in_data = pandas.read_excel(param['data']['opened'])
array = in_data[param['data']['text']].values

model = param['model']['model']

nlp = None
if model == 'sm':
    nlp = spacy.load('en_core_web_sm')
if model == 'md':
    nlp = spacy.load('en_core_web_md')
if model == 'lg':
    nlp = spacy.load('en_core_web_lg')
if nlp is None:
    raise ValueError("Insufficient vespine gas")

# Step 2. Make our data (with the vocabulary navigating columns)

start = True
start_len = 0
j = 0
result = []
columns = []
for y in array:
    doc = nlp(y)

    enha = {}
    for x in doc.ents:
        token = x.text
        code = x.label_

        if token in list(enha.keys()):
            if code not in enha[token]:
                enha[token].append(code)
        else:
            enha[token] = [code]

    # h = list(enha.keys())
    add_c = []

    for kk in enha.keys():
        for vv in enha[kk]:
            appie = "[{}]_['{}']".format(kk, vv)
            add_c.append(appie)
    outers = [z for z in add_c if z not in columns]
    columns = columns + outers
    h = outers

    if len(h) > 0:
        start_len = start_len + len(h)
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
columns = {j: 'R_SPC_{}{}'.format(code_, j) for j in data.columns.values}
pandas.DataFrame(data=result, columns=columns).to_excel('./data/gained.xlsx', index=False)
