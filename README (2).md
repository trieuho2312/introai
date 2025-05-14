# 🎵 Music Recommender System (Emotion & Lyrics Based)

A smart song recommendation system that suggests Vietnamese or international songs based on either user mood or song lyrics, using NLP and machine learning.



## 🚀 Features

- 🎯 **2 Modes of Recommendation**:
  - Gợi ý bài hát tương tự từ tên bài hát
  - Gợi ý bài hát dựa trên cảm xúc tự do người dùng nhập

- 🧠 **Emotion Classification** using Naive Bayes / SVM / Logistic Regression
- 📊 **Lyrics Vectorization** with TF-IDF + Cosine Similarity
- 🎵 **Spotify API** for cover image + play link
- ✍️ **Genius API** to fetch lyrics automatically for songs outside dataset
- 📦 Modular architecture for easy swapping of models/logic



## 📁 Project Structure


.
├── app/                 # Streamlit UI
│   └── app.py
├── recommender/         # Core logic and models
│   ├── emotion_model.py
│   ├── recommend.py
│   ├── preprocess.py
│   ├── spotify_api.py
│   ├── genius_api.py
│   └── ...
├── scripts/             # Scripts to prepare and train models
│   ├── train_emotion_model.py
│   ├── prepare_data.py
│   └── build_embedding_model.py
├── models/              # Stored model files (.pkl)
├── tests/               # Unit tests
├── requirements.txt
├── setup.py
└── .env                 # API Keys for Spotify & Genius
```



## 🔧 Installation


git clone https://github.com/yourname/music-recommender.git
cd music-recommender
#library
pip install -r requirements.txt

#setup.py
pip install -e .


Create a `.env` file and add:


SPOTIFY_CLIENT_ID=your_id
SPOTIFY_CLIENT_SECRET=your_secret
GENIUS_ACCESS_TOKEN=your_token




## ▶️ Run the App


streamlit run app/app.py




## 🧪 Emotion Classifier Training

python scripts/train_emotion_model.py


## 🛠 Build Vectorizer & Cosine Matrix

python scripts/prepare_data.py
python scripts/train_emotion_model.py

## 📌 Authors

> Ho Trieu - HUST

