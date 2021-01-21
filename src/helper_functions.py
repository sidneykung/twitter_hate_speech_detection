# importing relevant libraries
import numpy as np
import pandas as pd
import pickle

# NLP libraries
import re
import string
import nltk
from sklearn.feature_extraction import text 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# modeling libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import cross_val_score


## functions used in data_cleaning.ipynb

# tweet cleaning function
def clean_text_round1(text):
    '''Make text lowercase, remove punctuation, mentions, hashtags and words containing numbers.'''
    # make text lowercase
    text = text.lower()
    # removing text within brackets
    text = re.sub('\[.*?\]', '', text)
    # removing text within parentheses
    text = re.sub('\(.*?\)', '', text)
    # removing numbers
    text = re.sub('\w*\d\w*', '', text)
    # if there's more than 1 whitespace, then make it just 1
    text = re.sub('\s+', ' ', text)
    # if there's a new line, then make it a whitespace
    text = re.sub('\n', ' ', text)
    # removing any quotes
    text = re.sub('\"+', '', text)
    # removing &amp;
    text = re.sub('(\&amp\;)', '', text)
    # removing any usernames
    text = re.sub('(@[^\s]+)', '', text)
    # removing any hashtags
    text = re.sub('(#[^\s]+)', '', text)
    # remove `rt` for retweet
    text = re.sub('(rt)', '', text)
    # string.punctuation is a string of all punctuation marks
    # so this gets rid of all punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # getting rid of `httptco`
    text = re.sub('(httptco)', '', text)
    return text


## functions used in nlp_preprocessing.ipynb

def unfiltered_tokens(text):
    """tokenizing without removing stop words"""
    dirty_tokens = nltk.word_tokenize(text)
    return dirty_tokens

# tokenizing and removing stop words
def process_tweet(text):
    """tokenize text in each column and remove stop words"""
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
    return stopwords_removed 


## functions used for modeling process

# function to print all evaluation metrics
def evaluation(precision, recall, f1_score, f1_weighted):
    """prints out evaluation metrics for a model"""
    print('Testing Set Evaluation Metrics:')
    print('Precision: {:.4}'.format(precision))
    print('Recall: {:.4}'.format(recall))
    print('F1 Score: {:.4}'.format(f1_score))
    print('Weighted F1 Score: {:.4}'.format(f1_weighted))

# function to print training cross validation f1 stats
def train_cross_validation(model, X_train, y_train, metric, x):
    """prints cross-validation TRAINING metrics for a model"""
    scores = cross_val_score(model, X_train, y_train, scoring=metric, cv=x)
    print('Cross-Validation F1 Scores on Training Set:', scores)    
    print('\nMin: ', round(scores.min(), 6))
    print('Max: ', round(scores.max(), 6))
    print('Mean: ', round(scores.mean(), 6)) 
    print('Range: ', round(scores.max() - scores.min(), 6))

# determing out whether the model overfit or underfit
def model_fit(cv_train_metric, test_metric):
    if cv_train_metric > test_metric:
        models_fit = 'overfit'
    else:
        models_fit = 'underfit'
    return models_fit


## Doc2Vec Functions

# tokenizing specifically for Doc2Vec
def tokenize_text(text):
    """tokenize and remove stop words with NLTK"""
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def vec_for_learning(model, tagged_docs):    
    """final vector feature for classifier use"""
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors
