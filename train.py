import os
os.makedirs("model", exist_ok=True)

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import clean_text

def load_data(file):
    texts, labels = [], []
    with open(file, encoding='utf-8') as f:
        for line in f:
            text, emotion = line.strip().split(';')
            text = clean_text(text)
            label = 1 if emotion in ['sadness','fear','anger'] else 0
            texts.append(text)
            labels.append(label)
    return texts, labels

# Load data
X, y = load_data("data/train.txt")

# TF-IDF
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model
pickle.dump(model, open("model/model.pkl","wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl","wb"))

print("🔥 Model trained successfully!")