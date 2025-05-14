from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os
import joblib
from recommender.preprocess import clean_text

class EmotionClassifier:
    def __init__(self, model_path="models/emotion_model.pkl"):
        self.model_path = model_path
        self.model = None

    def train(self, data_path):
        df = pd.read_csv(data_path)
        df = df[['text', 'emotion']].dropna()
        df['cleaned'] = df['text'].apply(clean_text)
        X = df['cleaned']
        y = df['emotion']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", MultinomialNB())
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        joblib.dump(pipeline, self.model_path)
        print(f"✅ Saved model to {self.model_path}")

    def load(self):
        self.model = joblib.load(self.model_path)

    def predict(self, text):
        if not self.model:
            self.load()
        return self.model.predict([clean_text(text)])[0]