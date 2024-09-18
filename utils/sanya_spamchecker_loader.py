

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

def train_and_save_model():
    good_df = pd.read_excel('datasets/не_спам.xlsx')
    good_df = good_df.head(2400)
    good_df['labels'] = 0

    df_bad = pd.read_excel('datasets/спам.xlsx')
    df_bad = df_bad[['labels', 'text_ru']]
    df_bad = df_bad.query('labels == "spam"')
    df_bad['labels'] = 1

    df_result = pd.concat([df_bad, good_df])
    df_result['text_ru'] = df_result['text_ru'].apply(lambda x: str(x))
    df_result['text_ru'] = df_result['text_ru'].str.lower()
    
    spam_df = df_result[['labels', 'text_ru']]

    x_train, x_test, y_train, y_test = train_test_split(spam_df.text_ru, spam_df.labels)
    x_train.describe()

    # Находим слова, подсчитываем и собираем данные в матрицу
    cv = CountVectorizer()
    x_train_count = cv.fit_transform(x_train.values)
    x_train_count.toarray()

    # Тренируем модель
    model = MultinomialNB()
    model.fit(x_train_count, y_train)


    # Сохраняем модель и векторизатор
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(cv, 'count_vectorizer.pkl')
