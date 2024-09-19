import joblib

model = joblib.load('models/v2/spam_model.pkl')
cv = joblib.load('models/v2/count_vectorizer.pkl')




def sanya_spam_checkerv2(text):
    text = [text]
    email_spam_count = cv.transform(text)
    result = model.predict(email_spam_count)
    
    return result