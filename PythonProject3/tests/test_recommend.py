from recommender.emotion_model import EmotionClassifier

def test_predict_emotion():
    clf = EmotionClassifier()
    emotion = clf.predict("I miss you so much")
    assert emotion in ["happy", "sad", "love", "anger", "calm", "joy", "energetic"]