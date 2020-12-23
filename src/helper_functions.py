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

# tweet cleaning fucntion
def clean_text_round1(text):
    '''Make text lowercase, remove punctuation, mentions, hashtags and words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\(.*?\)', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\"+', '', text)
    text = re.sub('(\&amp\;)', '', text)
    text = re.sub('(@[^\s]+)', '', text)
    text = re.sub('(#[^\s]+)', '', text)
    text = re.sub('(rt)', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('(httptco)', '', text)
    return text

## functions used in nlp_preprocessing.ipynb

def unfiltered_tokens(text):
    """tokenizing withing removing stop words"""
    dirty_tokens = nltk.word_tokenize(text)
    return dirty_tokens

def process_tweet(text):
    """tokenize text in each column and remove stop words"""
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
    return stopwords_removed 

## functions used for modeling

# function to print evaluation metrics
def evaluation(precision, recall, f1, f1_weighted):
    """prints out evaluation metrics for a model"""
    print('Testing Evaluation Metrics:')
    print('Precision: {:.4}'.format(precision))
    print('Recall: {:.4}'.format(recall))
    print('F1 Score: {:.4}'.format(f1))
    print('Weighted F1 Score: {:.4}'.format(f1_weighted))

# function to print cross valudation
def cross_validation(model, X_train, y_train, metric, x):
    """prints cross-validation metrics for a model"""
    scores = cross_val_score(model, X_train, y_train, scoring=metric, cv=x)
    print('Cross-Validation F1 Scores:', scores)    
    print('\nMin: ', round(scores.min(), 6))
    print('Max: ', round(scores.max(), 6))
    print('Mean: ', round(scores.mean(), 6)) 
    print('Range: ', round(scores.max() - scores.min(), 6))

# Doc2Vec
def vec_for_learning(model, tagged_docs):    
    """final vector feature for classifier use"""
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors
