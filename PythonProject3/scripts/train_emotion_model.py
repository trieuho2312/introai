from recommender.emotion_model import EmotionClassifier

if __name__ == "__main__":
    clf = EmotionClassifier()
    clf.train("data/raw/spotify_dataset_10k.csv")