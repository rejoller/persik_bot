from utils.spamcheckerv4.spamchecker_loader import preprocess_text, delete_stopwords
import joblib
import os
from utils.spamcheckerv4.spamchecker_loader import vect_destination, models_destination, models_destination1




def predict_phrase_count_v(phrase, model):

    count_vectorizer = joblib.load(os.path.join(vect_destination, "count_vectorizer.pkl"))
    phrase = phrase.lower()
    preprocessed_phrase = preprocess_text(phrase) 
    cleaned_phrase = delete_stopwords(preprocessed_phrase)  
    joined_phrase = ' '.join(cleaned_phrase) 
    phrase_vect = count_vectorizer.transform([joined_phrase])
    prediction = model.predict(phrase_vect)
    
    return "Спам" if prediction[0] == 1 else "Не спам"


def predict_phrase_tf(phrase, model):

    tfidf_vectorizer = joblib.load(os.path.join(vect_destination, "tfidf_vectorizer.pkl"))
    phrase = phrase.lower()
    preprocessed_phrase = preprocess_text(phrase) 
    cleaned_phrase = delete_stopwords(preprocessed_phrase)  
    joined_phrase = ' '.join(cleaned_phrase)  
    phrase_vect = tfidf_vectorizer.transform([joined_phrase])
    prediction = model.predict(phrase_vect)
    
    return "Спам" if prediction[0] == 1 else "Не спам"






def ensemble_predict(phrase):

    nb_classifier = joblib.load(os.path.join(models_destination, "naive_bayes_model.pkl"))
    nb_classifier1 = joblib.load(os.path.join(models_destination, "naive_bayes_model1.pkl"))

    logreg = joblib.load(os.path.join(models_destination1, "logistic_regression_model.pkl"))
    logreg1 = joblib.load(os.path.join(models_destination1, "logistic_regression_model1.pkl"))
    
    log_reg_result1 = predict_phrase_tf(phrase, logreg1) 
    bayes_result1 = predict_phrase_tf(phrase, nb_classifier1)  
    log_reg_result = predict_phrase_count_v(phrase, logreg) 
    bayes_result = predict_phrase_count_v(phrase, nb_classifier) 

    results = [log_reg_result1, bayes_result1, log_reg_result, bayes_result]
    
    spam_votes = results.count("Спам")
    
    if spam_votes >= 2:  
        return 1
    else:
        return 0
