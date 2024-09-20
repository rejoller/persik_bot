import pandas as pd
from collections import Counter
import unidecode
from nltk.corpus import stopwords
import string
import re
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from icecream import ic
import logging
import joblib
import os



stop_words = set(stopwords.words('russian'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    tokens = [word for word in words if word not in stop_words]
    
    
    return tokens



def unidecoder(phrase):
    unidecoded_phrase = unidecode.unidecode(phrase)
    return unidecoded_phrase 


def train_and_save_model3():

    spam_df = pd.read_excel('datasets/v3/спам.xlsx', names=['text'])
    spam_df.head()
    spam_df.drop_duplicates(subset=['text'], inplace=True)

    no_spam_df = pd.read_excel('datasets/v3/не_спам.xlsx', names=['id', 'text'])
    no_spam_df = no_spam_df.head(30000)
    no_spam_df.drop(no_spam_df.index[0], inplace=True)

    no_spam_words = no_spam_df['text'].apply(preprocess_text).sum()
    spam_words = spam_df['text'].apply(preprocess_text).sum()

    unidecoded_nospam_words = []
    unidecoded_spam_words = []


    for word in no_spam_words:
        decoded_word = unidecoder(word)
        unidecoded_nospam_words.append(decoded_word)
        
    for word in spam_words:
        decoded_word = unidecoder(word)
        unidecoded_spam_words.append(decoded_word)
        
        
    unidecoded_nospam_words = [word for word in unidecoded_nospam_words if word not in ['', '-', 'A', 'V', '!!!', '{"type":', 'Vy', 'I', '.',
                                                                                        'g', 'let', 'd', 'p', 'rf', 'te', 'n', 't'
                                                                                        
                                                                                        'S', 'Da', '2.', '3.', '4.', '5.', 'eto', 'ul']]



    unidecoded_spam_words = [word for word in unidecoded_spam_words if word not in ['', 'you', 'your', 'a', '-', '--' 'A', 'for', 'c', 'to','is',
    'and',  'ot', 'k', 'Rs', 'of', 'Withdrawal', 'the',   'withdrawal',    'day',
    'ne','trx', 'b', 't', 'i', 'A', 'za', 'all', 'us', '+', 'do', 'p', 'y', 'u', 'C','[A][A][A][A]',
    'r', 'R', 'V', '!!!', '{"type":', 'Vy', 'I', '.', 'S', 'Da', '2.', '3.', '4.', '5.', 'eto', 'ul', 'na']]


    unidecodedf_nospam_df = pd.DataFrame(unidecoded_nospam_words, columns=['unidecoded_text'])
    unidecodedf_nospam_df = unidecodedf_nospam_df[unidecodedf_nospam_df['unidecoded_text'].apply(lambda x: len(x) >= 2)]
    unidecodedf_nospam_df['label'] = 0

    unidecodedf_spam_df = pd.DataFrame(unidecoded_spam_words, columns=['unidecoded_text'])
    unidecodedf_spam_df = unidecodedf_spam_df[unidecodedf_spam_df['unidecoded_text'].apply(lambda x: len(x) >= 2)]
    unidecodedf_spam_df['label'] = 1



    df = pd.concat([unidecodedf_nospam_df, unidecodedf_spam_df])


    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['unidecoded_text'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)*100
    
    model_dir = "models/v3/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Сохранение модели и векторизатора
    joblib.dump(nb_classifier, os.path.join(model_dir, "spam_model3.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer3.pkl"))


