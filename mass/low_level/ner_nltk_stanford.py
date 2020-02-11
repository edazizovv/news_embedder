import numpy
import pandas
from nltk.tag import StanfordNERTagger
# from nltk.tag.corenlp import CoreNLPNERTagger

import os

java_path = "C:/Program Files/Java/jdk-13.0.1/bin/java.exe"
os.environ['JAVAHOME'] = java_path
# TODO: configure these paths!
a1 = 'C:\\Users\\MainUser\\OneDrive\\RAMP-EXTERNAL\\IP-02\\OSTRTA\\models\\stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.crf.ser.gz'
a2 = 'C:\\Users\\MainUser\\OneDrive\\RAMP-EXTERNAL\\IP-02\\OSTRTA\\models\\stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.prop'

b = 'C:\\Users\\MainUser\\OneDrive\\RAMP-EXTERNAL\\IP-02\\OSTRTA\\models\\stanford-ner-2018-10-16\\stanford-ner.jar'

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

st = StanfordNERTagger(a1, b)
# st = CoreNLPNERTagger(a1, b)

# Step 2. Make our data (with the vocabulary navigating columns)

start = True
start_len = 0
j = 0
result = []
columns = []
for y in array:
    resu = st.tag(y.split())

    enha = {}
    for x in resu:
        token = x[0]
        code = x[1]

        if code != 'O':
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

    else:
        result.append(numpy.zeros(shape=(1, start_len)))


result = numpy.concatenate(result, axis=0)

data = pandas.DataFrame(data=result, columns=columns)
print('saving')
data.to_excel('./data/gained.xlsx', index=False)
