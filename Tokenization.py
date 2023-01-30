from nltk.tokenize import sent_tokenize, word_tokenize

sent = 'Hi I am Mahdiyar Abdollahi. Nice to meet you'
sent2 = 'I have never been to U.S.A'

print(word_tokenize(sent))
print(word_tokenize(sent2))
print(sent_tokenize(sent))