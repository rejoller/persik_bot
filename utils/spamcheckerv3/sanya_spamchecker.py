from utils.unidecoder import unidecoder
import joblib
from utils.spamcheckerv3.spamchecker_loader import preprocess_text



def spamchecker3(text):
    # Загрузка модели и векторизатора
    model = joblib.load("models/v2/spam_model3.pkl")
    vectorizer = joblib.load("models/v2/vectorizer3.pkl")

    # Предобработка введённой фразы
    processed_text = preprocess_text(text)
    processed_text_joined = ' '.join([unidecoder(word) for word in processed_text])

    # Векторизация
    text_vect = vectorizer.transform([processed_text_joined])

    # Прогнозирование
    prediction = model.predict(text_vect)
    
    return prediction