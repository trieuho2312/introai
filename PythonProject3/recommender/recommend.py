from recommender.data_manager import load_dataset, load_cosine, load_tfidf, load_model
from sklearn.metrics.pairwise import cosine_similarity
import logging

class SongRecommender:
    def __init__(self):
        self.df = load_dataset()
        self.cosine_sim = load_cosine()
        self.tfidf_matrix = load_tfidf()
        self.vectorizer = load_model("tfidf_vectorizer.pkl")

    def recommend(self, song_name, top_n=5):
        logging.info(f"🔍 Looking up song: {song_name}")
        idx_list = self.df[self.df['song'].str.lower() == song_name.lower()].index
        if len(idx_list) == 0:
            logging.warning("⚠️ Song not found in dataset.")
            return None
        idx = idx_list[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        song_indices = [i[0] for i in sim_scores]
        return self.df[['artist', 'song', 'emotion']].iloc[song_indices].reset_index(drop=True)

    def recommend_from_lyrics(self, lyrics, target_emotion=None, top_n=5):
        vec = self.vectorizer.transform([lyrics])

        if target_emotion:
            mask = self.df["emotion"].str.lower() == target_emotion.lower()
            df_filtered = self.df[mask].copy()
            tfidf_filtered = self.tfidf_matrix[mask.to_numpy()]

            if tfidf_filtered.shape[0] == 0:
                return None

            sims = cosine_similarity(vec, tfidf_filtered).flatten()
            top_indices = sims.argsort()[::-1][:top_n]

            return df_filtered.iloc[top_indices][['artist', 'song', 'emotion']].reset_index(drop=True)

        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1][:top_n]
        return self.df.iloc[top_indices][['artist', 'song', 'emotion']].reset_index(drop=True)
