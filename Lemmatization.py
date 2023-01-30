from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

machines = 'machines'
learning = 'learning'

stemmer = PorterStemmer()
lemmatize = WordNetLemmatizer()

#For EveryThing
print(machines, stemmer.stem(machines))
print(machines, stemmer.stem(learning))

#Only for Nouns
print(machines ,lemmatize.lemmatize(machines))
print(learning ,lemmatize.lemmatize(learning))