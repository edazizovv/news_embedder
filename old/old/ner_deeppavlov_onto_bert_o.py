import numpy
import pandas
from deeppavlov import configs, build_model

in_data = pandas.read_excel('./data/source.xlsx')
array = in_data['Text'].values

ner_model = build_model(configs.ner.ner_ontonotes_bert, download=True)  # done



# Step 1. We need a global vocabulary

general = {}
general_columns = []
for x in array:
    y = ner_model([x])

    enha = {}
    current_token_l = ''
    for j in range(len(y[1][0])):

        token = y[0][0][j]
        code = y[1][0][j]

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
    y = ner_model([x])

    enha = {}
    current_token_l = ''
    for j in range(len(y[1][0])):

        token = y[0][0][j]
        code = y[1][0][j]

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
