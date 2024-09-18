import joblib

model = joblib.load('spam_model.pkl')
cv = joblib.load('count_vectorizer.pkl')




def sanya_spam_checker(text):
    email_spam_count = cv.transform(text)
    result = model.predict(email_spam_count)
    
    return result