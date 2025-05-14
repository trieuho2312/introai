import os
import joblib
import pandas as pd

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model(obj, filename):
    ensure_dir(MODELS_DIR)
    joblib.dump(obj, os.path.join(MODELS_DIR, filename))

def load_model(filename):
    return joblib.load(os.path.join(MODELS_DIR, filename))

def load_dataset():
    return joblib.load(os.path.join(MODELS_DIR, "df_cleaned.pkl"))

def load_cosine():
    return joblib.load(os.path.join(MODELS_DIR, "cosine_sim.pkl"))

def load_tfidf():
    return joblib.load(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"))