import joblib
import os
from config import MODEL_DESTINATION_5, VECT_DESTINATION_5

vect = joblib.load(os.path.join(VECT_DESTINATION_5, "vectorizer.pkl"))
model = joblib.load(os.path.join(MODEL_DESTINATION_5, "stacking_model.pkl"))




async def spamchecker5(new_phrase):
    new_phrase = new_phrase.lower()
    new_phrase_vectorized = vect.transform([new_phrase])
    prediction = model.predict(new_phrase_vectorized)
    return prediction