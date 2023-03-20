import pandas as pd
import numpy as np
import spacy


nlp = spacy.load('en_core_web_md')
stopwords = spacy.lang.en.stop_words.STOP_WORDS
tag_types = {'noun': ['NN','NNP','NNS'],'verb': ['VB','VBD','VBG','VBN','VBP','VBZ']}

def read_file(filename):

    with open(filename) as f:
        lines = [x.split("\n")[0] for x in f.readlines()]
        for line in lines:
            doc = nlp(line)
            tags = [(token, token.tag_) for token in doc]
            print(tags)

def recomended_steps(sheetname):


    data = pd.read_excel('NLP test step intent.xlsx',sheet_name=sheetname)

    # data['stop_words_removed'] = data['Step / phrase'].apply(lambda x: " ".join([y for y in x.split() if y not in stopwords]))
    # data['steps'] = data['stop_words_removed'].apply(lambda x: nlp(x))
    data['steps'] = data['Step / phrase'].apply(lambda x: nlp(x))
    data['pos_tagged_steps'] = data['steps'].apply(lambda x: [(y,y.tag_) for y in x])
    data['intended_action'] = data['pos_tagged_steps'].apply(lambda x: [y[0] for y in x if y[1] in tag_types['verb']])
    data['intended_input'] = data['pos_tagged_steps'].apply(lambda x: [y[0] for y in x if y[1] in tag_types['noun']])
    data['intended_action'] = data['intended_action'].apply(lambda x: [nlp(''.join([token.text_with_ws for token in x]))] if len(x) > 1 else x)
    data['intended_action'] = data['intended_action'].apply(lambda x: x if len(x) > 0 else [nlp("NAN")])
    data['intended_input'] = data['intended_input'].apply(lambda x: [nlp(''.join([token.text_with_ws for token in x]))] if len(x) > 1 else x)
    data['intended_input'] = data['intended_input'].apply(lambda x: x if len(x) > 0 else [nlp("NAN")])

    data['Action intent'] = data['Action intent'].apply(lambda x: nlp(x))

    max_val_index = []
    similarity_matrix = np.zeros((data['intended_action'].size,data['Action intent'].size))
    for i in range(data['intended_action'].size):
        for j in range(data['Action intent'].size):
            similarity_matrix[i][j] = data['intended_action'].iloc[i][0].similarity(data['Action intent'].iloc[j][0])
        max_val_index.append(np.argmax(similarity_matrix[i]))

    print()
    print("step specified".ljust(20), "intended action".ljust(20), "closest match")
    print()
    for i in range(len(similarity_matrix[0])):
        step_specified = str(data['intended_action'].iloc[i][0])
        intended_action = str(data['Action intent'].iloc[i][0])
        closest_match = str(data['Action intent'].iloc[max_val_index[i]][0])
        print(step_specified.ljust(20),intended_action.ljust(20),closest_match)

    print(data['intended_input'])





