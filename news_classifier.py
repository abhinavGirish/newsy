from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import GaussianNB

import pandas as pd

# modified from Documents/building-features-text-data/code
# in this example, the label is Y, X is text from the dbpedia part

# http://localhost:8888/notebooks/Documents/building-features-text-data/code/12b-Stemmer_HashingVectorizer_NaiveBayesClassifier.ipynb

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)
    prec = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print("Length of testing data: ", len(y_test))
    print("accuracy_count : ", num_acc)
    print("accuracy_score : ", acc)
    print("precision_score : ", prec)
    print("recall_score : ", recall)


stemmer = SnowballStemmer('english')
analyzer = HashingVectorizer().build_analyzer()

stem_vectorizer = HashingVectorizer(n_features=2**10, norm='l2', analyzer=stemmed_words)

df = pd.read_csv('./archive/nyt-articles-2020.csv')

X = df['headline']
Y = df['n_comments']

feature_vector = stem_vectorizer.transform(X)

feature_vector.shape

X_dense = feature_vector.todense()

X_dense.shape

x_train, x_test, y_train, y_test = train_test_split(X_dense, Y, test_size = 0.2)

x_train.shape, x_test.shape

y_train.shape, y_test.shape

clf = GaussianNB().fit(x_train, y_train)

y_pred = clf.predict(x_test)
y_pred

summarize_classification(y_test, y_pred)