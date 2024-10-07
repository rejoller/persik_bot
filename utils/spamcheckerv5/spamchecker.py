import joblib
import re
from nltk.corpus import stopwords
import string
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from config import MODEL_DESTINATION_5, VECT_DESTINATION_5

vect = joblib.load(os.path.join(VECT_DESTINATION_5, "vectorizer.pkl"))
model = joblib.load(os.path.join(MODEL_DESTINATION_5, "stacking_model.pkl"))

stop_words = stopwords.words()
def remove_arabic(text):
    return re.sub(r'[\u0600-\u06FF]+', '', text)

def preprocess_text(text):

    if not isinstance(text, str):
        return []
    
    text = remove_arabic(text)
    
    # text = re.sub(r"[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+", "", text)
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    # text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)
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
    
    # stemmed_words = [stemmer.stem(word) for word in tokens]
    # ic(stemmed_words)
    # tokens = [word for word in words if word not in unidecoded_stop_words] 
    
    return tokens


def delete_stopwords(text):
    tokens = [word for word in text if word not in stop_words]  
    return tokens


def clean_text(text):
    cleaned_text = [re.sub(r'[,\[\]]', '', word) for word in text if word.strip()]
    return [word for word in cleaned_text if word]  # Убираем пустые строки


async def spamchecker5(new_phrase):
    new_phrase = new_phrase.lower()
    preprocessed_phrase = preprocess_text(new_phrase)  
    cleaned_phrase = delete_stopwords(preprocessed_phrase)
    joined_phrase = ' '.join(cleaned_phrase)
    
    
    
    
    new_phrase_vectorized = vect.transform([new_phrase])
    prediction = model.predict(new_phrase_vectorized)
    return prediction