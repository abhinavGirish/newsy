from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import GaussianNB

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

import pandas as pd

# modified from Documents/building-features-text-data/code
# in this example, the label is Y, X is text from the dbpedia part

# http://localhost:8888/notebooks/Documents/building-features-text-data/code/12b-Stemmer_HashingVectorizer_NaiveBayesClassifier.ipynb

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def summarize_classification(y_test, y_pred):
    #acc = accuracy_score(y_test, y_pred, normalize=True)
    #num_acc = accuracy_score(y_test, y_pred, normalize=False)
    #prec = precision_score(y_test, y_pred, average='weighted')
    #recall = recall_score(y_test, y_pred, average='weighted')
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mse = mean_squared_error(y_test, y_pred, squared=True)

    print("Length of testing data: ", len(y_test))
    print("root mean squared error: ", rmse)
    print("mean squared error: ", mse)


stemmer = SnowballStemmer('english')
analyzer = HashingVectorizer().build_analyzer()

df = pd.read_csv('./archive/nyt-articles-2020.csv')

X = df['headline']
Y = df['n_comments']

# frequency filtering
tokens = word_tokenize("\n".join(X.values))
freq = FreqDist(tokens)
frequent_words = []

for key, value in freq.items():
    if value >= 100:
        frequent_words.append(key.lower())

stop_words = text.ENGLISH_STOP_WORDS.union(frequent_words)

#stem_vectorizer = HashingVectorizer(n_features=2**10, norm='l2', analyzer=stemmed_words, ngram_range=(2,5)) #34 acc count
stem_vectorizer = HashingVectorizer(n_features=2**10, norm='l2', ngram_range=(2,3), stop_words=stop_words) #38 acc count

feature_vector = stem_vectorizer.transform(X)

feature_vector.shape

X_dense = feature_vector.todense()

X_dense.shape

x_train, x_test, y_train, y_test = train_test_split(X_dense, Y, test_size = 0.2)

x_train.shape, x_test.shape

y_train.shape, y_test.shape

clf = LinearRegression().fit(x_train, y_train)
#clf = GaussianNB().fit(x_train, y_train)

y_pred = clf.predict(x_test)
y_pred

print(y_test)
print(y_pred)

summarize_classification(y_test, y_pred)