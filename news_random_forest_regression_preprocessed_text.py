from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
import re
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import GaussianNB

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer

import pandas as pd

# modified from Documents/building-features-text-data/code
# in this example, the label is Y, X is text from the dbpedia part

# http://localhost:8888/notebooks/Documents/building-features-text-data/code/12b-Stemmer_HashingVectorizer_NaiveBayesClassifier.ipynb



def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

def summarize_classification(y_test, y_pred):
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mse = mean_squared_error(y_test, y_pred, squared=True)

    print("Length of testing data: ", len(y_test))
    print("root mean squared error: ", rmse)
    print("mean squared error: ", mse)




stemmer = SnowballStemmer('english')
analyzer = HashingVectorizer().build_analyzer()



#df = pd.read_csv('./archive/nyt-articles-2020.csv', encoding="ISO-8859-1")
df = pd.read_csv('./archive/modified.csv', encoding="ISO-8859-1")
#df = pd.read_csv('./archive/modified2.csv', encoding="ISO-8859-1")

df.dropna(subset = ['abstract'], inplace=True)

X = df['abstract']
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
    if sen == 1497:
        print("reached here")

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

stop_words = text.ENGLISH_STOP_WORDS.union(frequent_words)

#stem_vectorizer = HashingVectorizer(n_features=2**10, norm='l2', analyzer=stemmed_words, ngram_range=(2,5)) #34 acc count
#stem_vectorizer = HashingVectorizer(n_features=2**10, norm='l2',ngram_range=(2,3), stop_words=stop_words, analyzer=stemmed_words) #38 acc count
#stem_vectorizer = HashingVectorizer(n_features=2**10, norm='l2',ngram_range=(1,2), stop_words=stop_words)
#stem_vectorizer = HashingVectorizer(n_features=2**10, norm='l2',ngram_range=(1,2), stop_words=stop_words, analyzer=stemmed_words)
stem_vectorizer = HashingVectorizer(n_features=2**10, norm='l2', stop_words=stop_words, analyzer=stemmed_words)

feature_vector = stem_vectorizer.transform(documents)

feature_vector.shape

X_dense = feature_vector.todense()

X_dense.shape

print("Y modified: " + str(Y_modified.shape))
print("X : " + str(X_dense.shape))

print("Y type " + str(type(Y_modified)))
print("X type " + str(type(X_dense)))

x_train, x_test, y_train, y_test = train_test_split(X_dense, Y_modified['n_comments'], test_size = 0.2)

x_train.shape, x_test.shape

y_train.shape, y_test.shape

rgr = RandomForestRegressor(n_estimators=50)

print("random forest regressor created")

clf_rgr = rgr.fit(x_train, y_train)
y_pred_rgr = clf_rgr.predict(x_test)

print("y values predicted")

print(y_test)
print(y_pred_rgr)

summarize_classification(y_test, y_pred_rgr)