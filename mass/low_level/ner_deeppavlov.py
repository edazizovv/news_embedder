from deeppavlov import configs, build_model

which = args[0]

ner_model = None
if which == 'onto_bert_mult':
    ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)  # done
if which == 'onto_bert':
    ner_model = build_model(configs.ner.ner_ontonotes_bert, download=True)  # done
if which == 'onto':
    ner_model = build_model(configs.ner.ner_ontonotes, download=True)  # done
if which == 'conl_bert':
    ner_model = build_model(configs.ner.ner_conll2003_bert, download=True)  # done
if which == 'conl':
    ner_model = build_model(configs.ner.ner_conll2003, download=True)  # done
# if which == 'dstc2':  # deprecated
#     ner_model = build_model(configs.ner.ner_dstc2, download=True)  # done, but miss

if ner_model is None:
    raise ValueError("Insufficient vespine gas")

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