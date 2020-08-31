import pandas as pd
from sklearn import base
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer, HashingVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

nlp = spacy.load("en")

def tokenize_lemma(text):
    return [w.lemma_ for w in nlp(text)]

stop_words_lemma = set(tokenize_lemma(' '.join(STOP_WORDS)))

#Input the parameters for gridsearchcv to find the best results

param_features = {'vectorizer_name': TfidfVectorizer(),
                  # add vectorizer params as necessary
                  'max_features': [1000],
                  'ngram_range': [(1, 2)],
                  'stop_words': [stop_words_lemma.union({'regard', '-pron-'})],
                  'tokenizer': [tokenize_lemma],
                  'token_pattern': [None],
                  'max_df':[50, 100, 200],
                  'min_df': [3, 5]
                 }

param_model = {'model_name': SVC(),
               # add features as necessary for the model
               'C': [0.01, 0.1, 1, 10, 100],
               'kernel': ['linear', 'rbf', 'sigmoid'],
               'random_state': [42]
              }

df = pd.read_excel('training-Obama-Romney-tweets.xlsx', sheet_name='obama-cleaned', header=None, dtype={0: str})
df.columns = ['text', 'label']

#Shuffling the data
df = df.sample(frac=1, random_state=42)

#Removing nans
df = df[df['text'].notna()]
df = df[df['label'].notna()]

X = df[['text']]
y = df['label']

X = list(X['text'])
y = list(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


class clean_text(base.BaseEstimator, base.TransformerMixin):

    def __init__(self):
        pass  # We will need these in transform()

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self

    def transform(self, X):
        temp = []
        for row in X:
            line = row.lower().replace('<a>', ' ').replace('</a>', ' ').replace('<e>', ' ').replace('</e>', ' ')
            line = line.replace('#', 'hashtag')
            temp.append(line)
        return temp

ng_stem_tfidf = Pipeline([
                        ('clean', clean_text()),
                        ('tfidf', param_features['vectorizer_name']),
                        # Regressor
                        ('model', param_model['model_name'])
                    ])


parameters = {}
for row, i in enumerate(param_features.items()):
    if row > 0:
        parameters['tfidf__' + i[0]] = i[1]

for row, i in enumerate(param_model.items()):
    if row > 0:
        parameters['model__' + i[0]] = i[1]


best_predictor = GridSearchCV(ng_stem_tfidf, parameters, cv = 10, n_jobs = 1, verbose=1)
best_predictor.fit(X_train, y_train)

prediction_train = best_predictor.predict(X_train)
prediction_test = best_predictor.predict(X_test)

print('accuracy_train =', accuracy_score(prediction_train, y_train))
print('precision_train =', precision_score(prediction_train, y_train, average='macro'))
print('recall_train =', recall_score(prediction_train, y_train, average='macro'))
print('F1_Score_train =', f1_score(prediction_train, y_train, average='macro'))

print('accuracy_test =', accuracy_score(prediction_test, y_test))
print('precision_test =', precision_score(prediction_test, y_test, average='macro'))
print('recall_test =', recall_score(prediction_test, y_test, average='macro'))
print('F1_Score_test =', f1_score(prediction_test, y_test, average='macro'))

print('best parameters =', best_predictor.best_params_)


