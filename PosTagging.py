from nltk import word_tokenize, pos_tag, help

sent = 'Hi I am Mahdiyar Abdollahi. Nice to meet you'

word_token = word_tokenize(sent)

print(pos_tag(word_token))

print(help.upenn_tagset('VBP'))