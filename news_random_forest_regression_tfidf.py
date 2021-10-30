from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import RandomizedSearchCV
import re
import nltk
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer
from pprint import pprint

import pandas as pd

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def summarize_classification(y_test, y_pred):
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mse = mean_squared_error(y_test, y_pred, squared=True)
    print("Length of testing data: ", len(y_test))
    print("root mean squared error: ", rmse)
    print("mean squared error: ", mse)

nltk.download('wordnet')
stemmer = SnowballStemmer('english')
analyzer = HashingVectorizer().build_analyzer()

#df = pd.read_csv('./archive/nyt-articles-2020.csv', encoding="ISO-8859-1")
df = pd.read_csv('./archive/modified.csv', encoding="ISO-8859-1")
#df = pd.read_csv('./archive/modified2.csv', encoding="ISO-8859-1")

df.dropna(subset = ['abstract'], inplace=True)
#df.dropna(subset = ['headline'], inplace=True)

X = df['abstract']
#X = df['headline']
Y = df['n_comments']

print("X : ")
print(str(X))

print("Y : ")
print(str(type(Y)))

documents = []

# text preprocessing
print("length of X: " + str(len(X)))
print("shape of X: " + str(X.shape))
print("shape of Y: " + str(Y.shape))

Y_modified = pd.DataFrame()


for sen in range(0, len(X)):
    # Remove all the special characters

    doc = X.get(sen)

    if doc:
        document = re.sub(r'\W', ' ', str(doc))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        # document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

        #print("document: " + document)

        Y_modified = Y_modified.append({'n_comments': Y[sen]}, ignore_index=True)
    # else:
    #     print("row from Y that needs to be dropped: " + Y[sen])
    #     #Y = Y.drop(labels=sen, axis=0)

# frequency filtering


tokens = word_tokenize("\n".join(X.values))
freq = FreqDist(tokens)
frequent_words = []

for key, value in freq.items():
    if value >= 200:
        frequent_words.append(key.lower())

#stop_words = text.ENGLISH_STOP_WORDS.union(frequent_words)
stop_words = text.ENGLISH_STOP_WORDS

#stem_vectorizer = HashingVectorizer(n_features=2**10, norm='l2', stop_words=stop_words, analyzer=stemmed_words)
#processed_features = stem_vectorizer.transform(documents)

#vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, analyzer=stemmed_words, stop_words=stop_words)
#vectorizer = TfidfVectorizer(max_features=2500, analyzer=stemmed_words, max_df=0.8, stop_words=stop_words)
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=2500, analyzer=stemmed_words, max_df=0.8, stop_words=stop_words)
# vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=2500, max_df=0.8, stop_words=stop_words)
# vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=2500, max_df=0.95, stop_words=stop_words)

#vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=250, max_df=0.95, stop_words=stop_words)


# create the parameter grid:
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)


processed_features = vectorizer.fit_transform(documents)


processed_features.shape

X_dense = processed_features.todense()

X_dense.shape

print("Y modified: " + str(Y_modified.shape))
print("X : " + str(X_dense.shape))

print("Y type " + str(type(Y_modified)))
print("X type " + str(type(X_dense)))

x_train, x_test, y_train, y_test = train_test_split(X_dense, Y_modified['n_comments'], test_size = 0.2)

x_train.shape, x_test.shape

y_train.shape, y_test.shape

rgr = RandomForestRegressor(n_estimators=50)
rf_random = RandomizedSearchCV(estimator = rgr, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

print("random forest regressor created")

clf_rgr = rgr.fit(x_train, y_train)
y_pred_rgr = clf_rgr.predict(x_test)

print("y values predicted")

print(y_test)
print(y_pred_rgr)

summarize_classification(y_test, y_pred_rgr)