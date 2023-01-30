import spacy

sent = 'The book written by Hayden Liu in 2020 was sold at $30 in America'
nlp = spacy.load('en_core_web_sm')

tokens = nlp(sent)

for ent in tokens.ents:
    print(ent.text, ent.label_)