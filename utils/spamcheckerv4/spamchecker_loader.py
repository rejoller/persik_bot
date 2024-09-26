import logging
import pandas as pd
from collections import Counter
import unidecode
from nltk.corpus import stopwords
import nltk
import string
import re
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from icecream import ic
import os
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
lemm = WordNetLemmatizer()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from config import VECT_DESTINATION, MODELS_DESTINATION, MODELS_DESTINATION1





# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')

stop_words = stopwords.words()


spam_phrases = [
        "Дорогие друзья, сегодня наверное мой лучший день Я буквально час назад приняла участие в Акции от WB, оплатила комиссию и получила 88 тысяч рублей! https://dolofex.shop/?s=u3OtjmF9lS&p=1",
        "Kᴀк думаeшь, наделa я тpуcики или нет?",
        "Доброе утро👋 Пусть у каждого будет рассвет, с мирным небом и ласковым солнцем. Пусть не гаснет ваш внутренний свет и спокойствие в сердце вернётся! Чтобы свергнуть горы, необходимы вовсе не горы, а нескончаемая энергия, хорошее настроение и стимул. Здоровья вам и благополучия✨❤️",
        "Сегодня мой лучший день! В акции от Wildberries получилось выиграть 69 тысяч рублей) https://dolofex.shop/?s=R0FGE8IMYb&p=1",
        "Пройди по ссылке и забери 169 т . р. www.polniypizdec.ru",
        "Дᴘазни меʜя до тогo мoᴍентa, чтᴏбы я сама на тeбя ʜaбpоcилaсь…",
        "Вы получили налоговый возврат! Заберите свои деньги по ссылке: https://taxrefundnow.ru",
        "Поздравляем! Вы выиграли смартфон. Для получения приза подтвердите свой адрес: https://prizephone.shop",
        "Ваш аккаунт взломан. Для восстановления доступа перейдите по ссылке: https://securelogin.com/reset",
        "Купите билеты на концерт по суперцене! Всего 299 рублей! Забронируйте по ссылке: https://ticketdeal.com",
        "Ваш PayPal-аккаунт приостановлен. Подтвердите свою личность: https://paypalsecure.com",
        "Акция только сегодня! Получите 50% скидку на всё в нашем магазине: https://superdeals.ru",
        "💰 Эксклюзивное предложение! Инвестируйте в криптовалюту и получите доход до 300%! https://crypto-invest.com",
        "Ваш банк заблокировал карту. Для разблокировки перейдите на сайт: https://bankservice.ru/unblock",
        "🔥 Новые возможности для заработка! Успейте зарегистрироваться на платформе: https://profitday.com",
        "Получите бесплатный доступ к премиум-курсам! Акция действует 24 часа: https://learnnow.ru/free",
        "Эта инвестиционная платформа удвоила мои сбережения всего за неделю! Переходи по ссылке и узнай как: https://doubleyourmoney.com",
        "Kупи прямо сейчас и получи 90% скидку на следующий товар! Это предложение ограничено: https://flashsale.ru",
        "Поздравляем! Вы стали победителем в розыгрыше автомобиля. Заберите свой приз по ссылке: https://winacar.ru",
        "Срочное уведомление! Ваш аккаунт будет заблокирован через 24 часа, если не пройдете проверку: https://securelogin.ru",
        "Kликай на ссылку и получай мгновенный кэшбэк в 10 000 рублей! Акция заканчивается скоро: https://cashbacknow.ru",
        "Эксклюзивное предложение! Подпишитесь на VIP рассылку и получайте еженедельные бонусы: https://vipoffers.ru",
        "Ваша подписка на стриминговый сервис приостановлена. Активируйте её снова по ссылке: https://streamingfix.com",
        "Получите свой подарочный сертификат на 1000 рублей! Подтвердите свой адрес: https://giftcardnow.ru",
        "Поздравляем, вы выиграли поездку на двоих! Заберите свой билет по ссылке: https://travelwin.ru"
    ]


def remove_arabic(text):
    return re.sub(r'[\u0600-\u06FF]+', '', text)

def delete_stopwords(text):
    tokens = [word for word in text if word not in stop_words]  
    return tokens

def clean_text(text):
    cleaned_text = [re.sub(r'[,\[\]]', '', word) for word in text if word.strip()]
    return [word for word in cleaned_text if word]  # Убираем пустые строки




def preprocess_text(text):

    if not isinstance(text, str):
        return []
    text = remove_arabic(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", "", text)

    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d', '', text)
    text = text.lower() 
    tokens = word_tokenize(text, language='russian')

    return tokens

def train_and_save_model4():
    try:
        spam_df = pd.read_excel('спам.xlsx', names=['text'])
        spam_df.drop_duplicates(subset=['text'], inplace=True)
        
        
        no_spam_df = pd.read_excel('не_спам.xlsx', names=['id', 'text'])
        no_spam_df = no_spam_df.head(30000)
        no_spam_df.drop(no_spam_df.index[0], inplace=True)
        
        spam_df['preprocessed_text'] = spam_df['text'].apply(preprocess_text)
        no_spam_df['preprocessed_text'] = no_spam_df['text'].apply(preprocess_text)
        
        no_spam_df['cleaned_text'] = no_spam_df['preprocessed_text'].apply(delete_stopwords)
        spam_df['cleaned_text']= spam_df['preprocessed_text'].apply(delete_stopwords)
        
        spam_df['label'] = 1
        no_spam_df['label'] = 0
        
        df = pd.concat([spam_df, no_spam_df])
        df['cleaned_text'].apply(clean_text)
        df['joined_text'] = df['cleaned_text'].apply(lambda x: ' '.join(x) if x else '')
        df = df.query('joined_text != ""')
        
        
        
        df_unidecoded = df[['joined_text', 'label']]
        

        spam_phrases_extended = spam_phrases * 10

        spam_labels = [1] * len(spam_phrases_extended)
        new_spam_data = pd.DataFrame({
            'joined_text': spam_phrases_extended,
            'label': spam_labels
        })

        df_train_updated = pd.concat([df_unidecoded, new_spam_data], ignore_index=True)
        
        
        count_vectorizer = CountVectorizer(ngram_range=(1, 3))
        X = count_vectorizer.fit_transform(df_train_updated['joined_text'])
        y = df_train_updated['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        X1 = tfidf_vectorizer.fit_transform(df_train_updated['joined_text'])
        y1 = df_train_updated['label']
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
        
        
        
        

        nb_classifier = MultinomialNB(alpha=0.5)
        nb_classifier.fit(X_train, y_train)


        y_pred = nb_classifier.predict(X_test)


        logreg = LogisticRegression(C=10, solver='liblinear')
        logreg.fit(X_train, y_train)
        y_pred_logreg = logreg.predict(X_test)


        nb_classifier1 = MultinomialNB(alpha=0.6618)
        nb_classifier1.fit(X_train1, y_train1)

        y_pred1 = nb_classifier1.predict(X_test1)


        logreg1 = LogisticRegression(C=10, solver='liblinear')
        logreg1.fit(X_train1, y_train1)

        y_pred_logreg1 = logreg1.predict(X_test1)
        
        

        if not os.path.exists(VECT_DESTINATION):
            os.makedirs(VECT_DESTINATION)
        
        
        joblib.dump(tfidf_vectorizer, os.path.join(VECT_DESTINATION, "tfidf_vectorizer.pkl"))
        joblib.dump(count_vectorizer, os.path.join(VECT_DESTINATION, "count_vectorizer.pkl"))


        if not os.path.exists(MODELS_DESTINATION):
            os.makedirs(MODELS_DESTINATION)
        
        joblib.dump(nb_classifier, os.path.join(MODELS_DESTINATION, "naive_bayes_model.pkl"))
        joblib.dump(nb_classifier1, os.path.join(MODELS_DESTINATION, "naive_bayes_model1.pkl"))
        
        MODELS_DESTINATION1
        if not os.path.exists(MODELS_DESTINATION1):
            os.makedirs(MODELS_DESTINATION1)
        
        joblib.dump(logreg, os.path.join(MODELS_DESTINATION1, "logistic_regression_model.pkl"))
        joblib.dump(logreg1, os.path.join(MODELS_DESTINATION1, "logistic_regression_model1.pkl"))
        logging.info(f'модель натренирована и сохранена в {VECT_DESTINATION} {MODELS_DESTINATION1} {MODELS_DESTINATION1}')
    except Exception as e:
        logging.error(f"ошибка при обучении и сохранении модели: {e}")