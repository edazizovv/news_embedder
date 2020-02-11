import numpy
import pandas
from deeppavlov import configs, build_model

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

ner_model = build_model(configs.ner.ner_ontonotes_bert, download=True)  # done

# Step 2. Make our data (with the vocabulary navigating columns)

start = True
start_len = 0
result = []
columns = []
for x in array:
    y = ner_model([x])

    enha = {}
    current_token_l = ''
    for jj in range(len(y[1][0])):

        token = y[0][0][jj]
        code = y[1][0][jj]

        if code != 'O':

            code_mark = code[0]
            code_label = code[2:]

            if code_mark == 'B':
                current_token_l = token

            if code_mark == 'I':
                del enha[current_token_l]
                current_token_l = current_token_l + ' ' + token

            if current_token_l in list(enha.keys()):
                if code_label not in enha[current_token_l]:
                    enha[current_token_l].append(code_label)
            else:
                enha[current_token_l] = [code_label]

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
