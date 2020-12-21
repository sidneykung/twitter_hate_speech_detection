""" 


    PLEASE READ FIRST:

    This is a web app created with Streamlit to host this project. 

    If you use any of this code, please credit with a link to my website:
    https://www.sidneykung.com/


""" 

# importing python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn import svm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# creating page sections
site_header = st.beta_container()
business_context = st.beta_container()
tweet_input = st.beta_container()
model_results = st.beta_container()
sentimet_analysis = st.beta_container()
# model_training = st.beta_container()

with site_header:
    st.title('Twitter Hate Speech Detection')
    st.write("""
    Created by [Sidney Kung](https://www.sidneykung.com/)
    
    This project aims to **automate content moderation** to identify hate speech using **machine learning binary classification algorithms.** 
    
    Baseline models included Random Forest, Naive Bayes, Logistic Regression and Support Vector Machine (SVM). **The final model was a Linear SVM model** with an F1 of 0.3955 and Recall (TPR) of 0.4373. 
        
    Check out the project repository [here](https://github.com/sidneykung/twitter_hate_speech_detection).
    """)

with business_context:
    st.header('Business Context')
    st.write("""
    
    **Human content moderation exploits people by consistently traumatizing and underpaying them.** In 2019, an [article](https://www.theverge.com/2019/6/19/18681845/facebook-moderator-interviews-video-trauma-ptsd-cognizant-tampa) on The Verge exposed the extensive list of horrific working conditions that employees faced at Cognizant, which was Facebook’s primary moderation contractor. Unfortunately, **every major tech company**, including **Twitter**, uses human moderators to some extent, both domestically and overseas.
    
    Hate speech is defined as **abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion or sexual orientation.**  Usually, the difference between hate speech and offensive language comes down to subtle context or diction.
    
    The data for this project was sourced from a Cornell University [study](https://github.com/t-davidson/hate-speech-and-offensive-language) titled *Automated Hate Speech Detection and the Problem of Offensive Language*.

    """)
    st.markdown("---")

with tweet_input:
    st.header('Is Your Tweet Considered Hate Speech?')
    user_text = st.text_input('Enter Tweet', max_chars=280) # setting input as user_text

with model_results:
    st.header('Results')
    st.write("""This section will output Linear SVM model prediction.""")

with sentimet_analysis:
    st.header('Sentiment Analysis with VADER')
    
    # instantiating VADER sentiment analyzer
    sid_obj = SentimentIntensityAnalyzer() 
    # the object outputs the scores into a dict
    sentiment_dict = sid_obj.polarity_scores(user_text) 

    if sentiment_dict['compound'] >= 0.05 : 
        category = ("**Positive ✅**")
    elif sentiment_dict['compound'] <= - 0.05 : 
        category = ("**Negative 🚫**") 
    else : 
        category = ("**Neutral ☑️**")

    # explaining VADER
    st.write("""*VADER is a lexicon designed for scoring social media. More information can be found [here](https://github.com/cjhutto/vaderSentiment).*""")
    # spacer
    st.text('')

# score breakdown section with columns
col1, col2 = st.beta_columns(2)
with col1:
    # printing category
    st.write("Your Tweet is Rated as", category) 
    # printing overall compound score
    st.write("**Compound Score**: ", sentiment_dict['compound'])
    # printing overall compound score
    st.write("**Sentiment Breakdown:**") 
    st.write(sentiment_dict['neg']*100, "% Negative") 
    st.write(sentiment_dict['neu']*100, "% Neutral") 
    st.write(sentiment_dict['pos']*100, "% Positive") 
with col2:
    sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
    st.bar_chart(sentiment_graph) 


# with model_training:
#     st.header('Model training')
#     st.text('In this section you can select the hyperparameters!')