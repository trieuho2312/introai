import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommender.data_manager import ensure_dir

DATA_PATH = "models/df_cleaned.pkl"
TFIDF_PATH = "models/tfidf_matrix.pkl"
COSINE_PATH = "models/cosine_sim.pkl"

if __name__ == "__main__":
    df = joblib.load(DATA_PATH)
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    ensure_dir("models")
    joblib.dump(tfidf_matrix, TFIDF_PATH)
    joblib.dump(cosine_sim, COSINE_PATH)

    print(f"✅ TF-IDF and cosine matrix saved.")