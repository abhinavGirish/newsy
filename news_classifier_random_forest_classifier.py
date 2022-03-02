from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
from sklearn import metrics

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

    # alternative error computation
    errors = abs(y_pred - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print("Model Performance")
    print("Average Error: {:0.4f} degrees.".format(np.mean(errors)))
    print('Accuracy = {:0.2f}%'.format(accuracy))


nltk.download('wordnet')
stemmer = SnowballStemmer('english')
analyzer = HashingVectorizer().build_analyzer()

df = pd.read_csv('./archive/modified.csv', encoding="ISO-8859-1")

df.dropna(subset = ['abstract','headline'], inplace=True)

X = df['abstract']
X_headline = df['headline']
Y = df['clickbait_category_4']

documents = []

# text preprocessing

Y_modified = pd.DataFrame()


for sen in range(0, len(X)):
    # Remove all the special characters

    headline = X_headline.get(sen)
    abstract = X.get(sen)
    """if headline is None:
        headline = ""

    if abstract is None:
        abstract = "" """

    #doc = headline + " : " + X.get(sen)

    if not(headline is None) and not(abstract is None):

        doc = headline + " : " + abstract
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

        Y_modified = Y_modified.append({'clickbait_category_4': Y[sen]}, ignore_index=True)


# frequency filtering


tokens = word_tokenize("\n".join(X.values))
freq = FreqDist(tokens)
frequent_words = []

for key, value in freq.items():
    if value >= 200:
        frequent_words.append(key.lower())

stop_words = text.ENGLISH_STOP_WORDS

vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=2500, analyzer=stemmed_words, max_df=0.8, stop_words=stop_words)

# create the parameter grid:
"""n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

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

pprint(random_grid)"""


processed_features = vectorizer.fit_transform(documents)

processed_features.shape

X_dense = processed_features.todense()

X_dense.shape

x_train, x_test, y_train, y_test = train_test_split(X_dense, Y_modified['clickbait_category_4'], test_size = 0.2) # old was 0.2

x_train.shape, x_test.shape

y_train.shape, y_test.shape

#rgr = RandomForestRegressor()
#rf_random = RandomizedSearchCV(estimator = rgr, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("ACCURACY OF THE MODEL : ", metrics.accuracy_score(y_test, y_pred))