import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB




def train_and_save_model2():

    no_spam_df = pd.read_excel('datasets/v2/не_спам.xlsx')
    spam_df = pd.read_excel('datasets/v2/2024_сообщения.xlsx')

    no_spam_df = pd.read_excel('datasets/v2/не_спам.xlsx')
    spam_df = pd.read_excel('datasets/v2/2024_сообщения.xlsx')

    

    df = pd.concat([no_spam_df, spam_df])


    stop_words = set(stopwords.words('russian')) 

    spam_upsampled = resample(df[df['label'] == 1],
                            replace=True, 
                            n_samples=len(df[df['label'] == 0]), 
                            random_state=42)

    df_balanced = pd.concat([df[df['label'] == 0], spam_upsampled])
    df_balanced['cleaned'] = df_balanced['cleaned'].fillna('')

    X_train, X_test, y_train, y_test = train_test_split(df_balanced['cleaned'], df_balanced['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)


    X_train_dense = X_train_tfidf.toarray()

    X_test_dense = X_test_tfidf.toarray()

    gnb = GaussianNB()
    gnb.fit(X_train_dense, y_train)

    joblib.dump(gnb, 'models/v2/spam_model.pkl')
    joblib.dump(vectorizer, 'models/v2/tfidf_vectorizer.pkl')


