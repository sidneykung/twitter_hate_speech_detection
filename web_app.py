""" 


    PLEASE NOTE:

    This is an interactive web app created with StreamLit.

    It's hosted on Heroku here:
    https://hate-speech-predictor.herokuapp.com/

    If you use any of this code, please credit with a link to my website:
    https://www.sidneykung.com/


""" 

# importing relevant python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
# modeling
from sklearn import svm
# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# creating page sections
site_header = st.beta_container()
business_context = st.beta_container()
data_desc = st.beta_container()
performance = st.beta_container()
tweet_input = st.beta_container()
model_results = st.beta_container()
sentiment_analysis = st.beta_container()
contact = st.beta_container()

with site_header:
    st.title('Twitter Hate Speech Detection')
    st.write("""
    Created by [Sidney Kung](https://www.sidneykung.com/)
    
    This project aims to **automate content moderation** to identify hate speech using **machine learning binary classification algorithms.** 
    
    Baseline models included Random Forest, Naive Bayes, Logistic Regression and Support Vector Machine (SVM). The final model was a **Logistic Regression** model that used Count Vectorization for feature engineering. It produced an F1 of 0.3958 and Recall (TPR) of 0.624.  
        
    Check out the project repository [here](https://github.com/sidneykung/twitter_hate_speech_detection).
    """)

with business_context:
    st.header('The Problem of Content Moderation')
    st.write("""
    
    **Human content moderation exploits people by consistently traumatizing and underpaying them.** In 2019, an [article](https://www.theverge.com/2019/6/19/18681845/facebook-moderator-interviews-video-trauma-ptsd-cognizant-tampa) on The Verge exposed the extensive list of horrific working conditions that employees faced at Cognizant, which was Facebook’s primary moderation contractor. Unfortunately, **every major tech company**, including **Twitter**, uses human moderators to some extent, both domestically and overseas.
    
    Hate speech is defined as **abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion or sexual orientation.**  Usually, the difference between hate speech and offensive language comes down to subtle context or diction.
    
    """)

with data_desc:
    understanding, venn = st.beta_columns(2)
    with understanding:
        st.text('')
        st.write("""
        The **data** for this project was sourced from a Cornell University [study](https://github.com/t-davidson/hate-speech-and-offensive-language) titled *Automated Hate Speech Detection and the Problem of Offensive Language*.
        
        The `.csv` file has **24,802 rows** where **6% of the tweets were labeled as "Hate Speech".**

        Each tweet's label was voted on by crowdsource and determined by majority rules.
        """)
    with venn:
        st.image(Image.open('visualizations/word_venn.png'), width = 400)

with performance:
    description, conf_matrix = st.beta_columns(2)
    with description:
        st.header('Final Model Performance')
        st.write("""
        These scores are indicative of the two major roadblocks of the project:
        - The massive class imbalance of the dataset
        - The model's inability to identify what constitutes as hate speech
        """)
    with conf_matrix:
        st.image(Image.open('visualizations/normalized_log_reg_countvec_matrix.png'), width = 400)

with tweet_input:
    st.header('Is Your Tweet Considered Hate Speech?')
    st.write("""*Please note that this prediction is based on how the model was trained, so it may not be an accurate representation.*""")
    # user input here
    user_text = st.text_input('Enter Tweet', max_chars=280) # setting input as user_text

with model_results:    
    st.subheader('Prediction:')
    if user_text:
    # processing user_text
        # removing punctuation
        user_text = re.sub('[%s]' % re.escape(string.punctuation), '', user_text)
        # tokenizing
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(user_text)
        # removing stop words
        stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
        # taking root word
        lemmatizer = WordNetLemmatizer() 
        lemmatized_output = []
        for word in stopwords_removed:
            lemmatized_output.append(lemmatizer.lemmatize(word))

        # instantiating count vectorizor
        count = CountVectorizer(stop_words=stop_words)
        X_train = pickle.load(open('pickle/X_train_2.pkl', 'rb'))
        X_test = lemmatized_output
        X_train_count = count.fit_transform(X_train)
        X_test_count = count.transform(X_test)

        # loading in model
        final_model = pickle.load(open('pickle/final_log_reg_count_model.pkl', 'rb'))

        # apply model to make predictions
        prediction = final_model.predict(X_test_count[0])

        if prediction == 0:
            st.subheader('**Not Hate Speech**')
        else:
            st.subheader('**Hate Speech**')
        st.text('')

with sentiment_analysis:
    if user_text:
        st.header('Sentiment Analysis with VADER')
        
        # explaining VADER
        st.write("""*VADER is a lexicon designed for scoring social media. More information can be found [here](https://github.com/cjhutto/vaderSentiment).*""")
        # spacer
        st.text('')
    
        # instantiating VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer() 
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text) 
        if sentiment_dict['compound'] >= 0.05 : 
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05 : 
            category = ("**Negative 🚫**") 
        else : 
            category = ("**Neutral ☑️**")

        # score breakdown section with columns
        breakdown, graph = st.beta_columns(2)
        with breakdown:
            # printing category
            st.write("Your Tweet is rated as", category) 
            # printing overall compound score
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            # printing overall compound score
            st.write("**Polarity Breakdown:**") 
            st.write(sentiment_dict['neg']*100, "% Negative") 
            st.write(sentiment_dict['neu']*100, "% Neutral") 
            st.write(sentiment_dict['pos']*100, "% Positive") 
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph) 

with contact:
    st.markdown("---")
    st.header('For More Information')
    st.text('')
    st.write("""

    **Check out the project repository [here](https://github.com/sidneykung/twitter_hate_speech_detection).**

    Contact Sidney Kung via [sidneyjkung@gmail.com](mailto:sidneyjkung@gmail.com).
    """)

    st.subheader("Let's Connect!")
    st.write("""
    
    [LinkedIn](https://www.linkedin.com/in/sidneykung/) | [Github](https://github.com/sidneykung)  |  [Medium](https://medium.com/@sidneykung)  |  [Twitter](https://twitter.com/sidney_k98)


    """)
