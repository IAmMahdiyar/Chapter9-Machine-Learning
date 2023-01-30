import spacy

sent = 'Hi I am Mahdiyar Abdollahi. Nice to meet you'
nlp = spacy.load('en_core_web_sm')

tokens = nlp(sent)

for token in tokens:
    print(token)