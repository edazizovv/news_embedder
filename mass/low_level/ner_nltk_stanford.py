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


# Step 1. We need a global vocabulary

general = {}
general_columns = []
for x in array:
    resu = st.tag(x.split())

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

    for key in enha.keys():
        for value in enha[key]:
            if key in general.keys():
                if value in general[key].keys():
                    pass
                else:
                    named = '[{}]_{}'.format(key, value)
                    general[key][value] = named
                    general_columns.append(named)
            else:
                named = '[{}]_{}'.format(key, value)
                general[key] = {}
                general[key][value] = named
                general_columns.append(named)

# Step 2. Make our data (with the vocabulary navigating columns)

result = []
for x in array:
    resu = st.tag(x.split())

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

    values = numpy.zeros(shape=(1, len(general_columns)))
    for key in enha.keys():
        for value in enha[key]:
            ix = general_columns.index(general[key][value])
            values[0, ix] = 1
    result.append(values)


result = numpy.concatenate(result, axis=0)



data = pandas.DataFrame(data=result, columns=general_columns)
print('saved')
data.to_excel('./data/gained.xlsx', index=False)
