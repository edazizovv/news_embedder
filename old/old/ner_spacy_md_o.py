import numpy
import pandas
import spacy

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

nlp = spacy.load('en_core_web_md')

# Step 1. We need a global vocabulary

general = {}
general_columns = []
for x in array:
    doc = nlp(x)

    enha = {}
    for x in doc.ents:
        token = x.text
        code = x.label_

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
    doc = nlp(x)

    enha = {}
    for x in doc.ents:
        token = x.text
        code = x.label_

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
