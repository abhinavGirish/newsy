import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.feature_extraction import text
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
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

"""
model_fn
    model_dir: (sting) specifies location of saved model

This function is used by AWS Sagemaker to load the model for deployment.
It does this by simply loading the model that was saved at the end of the
__main__ training block above and returning it to be used by the predict_fn
function below.
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: the body of the request sent to the model. The type can vary.
    request_content_type: (string) specifies the format/variable type of the request

This function is used by AWS Sagemaker to format a request body that is sent to 
the deployed model.
In order to do this, we must transform the request body into a numpy array and
return that array to be used by the predict_fn function below.

Note: Oftentimes, you will have multiple cases in order to
handle various request_content_types. Howver, in this simple case, we are 
only going to accept text/csv and raise an error for all other formats.
"""
def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        samples = []
        for r in request_body.split('|'):
            samples.append(list(map(float,r.split(','))))
        return np.array(samples)
    else:
        raise ValueError("Thie model only supports text/csv input")

"""
predict_fn
    input_data: (numpy array) returned array from input_fn above 
    model (sklearn model) returned model loaded from model_fn above

This function is used by AWS Sagemaker to make the prediction on the data
formatted by the input_fn above using the trained model.
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: (string) the content type the endpoint expects to be returned

This function reformats the predictions returned from predict_fn to the final
format that will be returned as the API call response.

Note: While we don't use content_type in this example, oftentimes you will use
that argument to handle different expected return types.
"""
def output_fn(prediction, content_type):
    return '|'.join([t for t in prediction])

if __name__ =='__main__':
    # Create a parser object to collect the environment variables that are in the
    # default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    # Load data from the location specified by args.train (In this case, an S3 bucket).
    nltk.download('wordnet')
    nltk.download('punkt')

    stemmer = SnowballStemmer('english')
    analyzer = HashingVectorizer().build_analyzer()

    #df = pd.read_csv('./archive/modified.csv', encoding="ISO-8859-1")

    df = pd.read_csv(os.path.join(args.train, 'modified.csv'), index_col=0, engine="python")

    df.dropna(subset = ['abstract','headline'], inplace=True)

    X = df['abstract']
    X_headline = df['headline']
    X_section = df['section']
    Y = df['clickbait_category_4']

    documents = []

    # text preprocessing

    Y_modified = pd.DataFrame()


    for sen in range(0, len(X)):
        # Remove all the special characters

        headline = X_headline.get(sen)
        abstract = X.get(sen)
        section = X_section.get(sen)

        if not(headline is None) and not(abstract is None) and not(section is None):

            doc = section + " : " + headline + " : " + abstract
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

    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=2500, analyzer=stemmed_words , stop_words=stop_words)

    # create the parameter grid:

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
    # 'n_estimators': n_estimators,

    random_grid = {
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

    x_train, x_test, y_train, y_test = train_test_split(X_dense, Y_modified['clickbait_category_4'], test_size = 0.2)

    x_train.shape, x_test.shape

    y_train.shape, y_test.shape

    clf = RandomForestClassifier(n_estimators = 100)

    model = HalvingGridSearchCV(estimator=clf, param_grid = random_grid, cv=3, factor=2,
                                            resource='n_estimators',max_resources=30).fit(x_train, y_train)

    #Save the model to the location specified by args.model_dir
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))