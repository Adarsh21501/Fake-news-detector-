import os
import re
import pickle
import nltk
import tensorflow as tf
from django.shortcuts import render
from .forms import NewsForm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Paths to model and tokenizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_files/fake_news_model.h5')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), 'model_files/tokenizer.pkl')
BEST_THRESHOLD=os.path.join(os.path.dirname(__file__), 'model_files/best_threshold.pkl')

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

MAX_SEQUENCE_LENGTH = 200   
BEST_THRESHOLD = 0.5

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def predict_news(request):
    result = None
    confidence = None

    if request.method == 'POST':
        form = NewsForm(request.POST)
        if form.is_valid():
            raw_text = form.cleaned_data['news_text']
            cleaned = preprocess_text(raw_text)
            print(f"Cleaned text: {cleaned}")

            seq = tokenizer.texts_to_sequences([cleaned])
            print(f"Tokenized sequence: {seq}")

            padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
            print(f"Padded sequence: {padded}")

            prob = model.predict(padded)[0][0]
            print(f"Model output (prob): {prob}")

            result = "Real" if prob >= BEST_THRESHOLD else "Fake"
            confidence = f"{prob:.3f} (threshold={BEST_THRESHOLD:.2f})"
    else:
        form = NewsForm()

    return render(request, 'news_form.html', {'form': form, 'result': result, 'confidence': confidence})
