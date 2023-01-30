from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import names
from sklearn.manifold import TSNE

categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']
groups = fetch_20newsgroups(categories=categories_3)

print(groups.keys())

#Target Range
print(np.unique(groups.target))

#Shows Distribution
sns.distplot(groups.target)
plt.show()

#Shows Top 500 Repeated Words
count_vector = CountVectorizer(max_features=500, stop_words='english')
count_vector.fit_transform(groups.data)
print(count_vector.get_feature_names_out())

#Data Cleaning
print('Data Cleaning')
porter = PorterStemmer()
all_names = names.words()
data_cleaned = []
for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(porter.stem(word)
                        for word in doc.split()
                        if word.isalpha() and
                        word not in all_names)
    data_cleaned.append(doc_cleaned)

#Count Clean Data
print('Fitting And Transform with Clean Data')
data_clean_count = count_vector.fit_transform(data_cleaned)

print('Fitting TSNE Model')
tsne = TSNE(n_components=2, perplexity=40, learning_rate=500)
data_tsne = tsne.fit_transform(data_clean_count.toarray())
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups.target)
plt.show()