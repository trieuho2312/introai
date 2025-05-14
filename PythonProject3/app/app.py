import streamlit as st
from recommender.recommend import SongRecommender
from recommender.emotion_model import EmotionClassifier
from recommender.spotify_api import search_track
from recommender.genius_api import get_lyrics

st.set_page_config(page_title="🎵 Music Recommender", layout="wide")

st.title("🎵 Music Recommender System")
st.markdown("---")

option = st.radio("🎯 Chọn chức năng:", (
    "Gợi ý bài hát tương tự",
    "Tìm bài hát theo lời nhập tự do"
), horizontal=False)

recommender = SongRecommender()
clf = EmotionClassifier()

if option == "Gợi ý bài hát tương tự":
    song_name = st.text_input("🎯 Nhập tên bài hát bạn đang nghe hoặc muốn gợi ý:")
    if song_name:
        df = recommender.df
        matched = df[df['song'].str.lower() == song_name.lower()]
        results = None

        if not matched.empty:
            st.success("✅ Bài hát có trong hệ thống. Gợi ý bài tương tự:")
            results = recommender.recommend(song_name)
        else:
            st.warning("🔍 Bài hát không có sẵn trong hệ thống. Đang tìm lyrics...")
            sp_result = search_track(song_name)
            if sp_result:
                st.markdown(f"**🎶 Đã tìm thấy:** `{sp_result['song']}` - *{sp_result['artist']}*")
                st.image(sp_result['image'], width=300)
                st.markdown(f"[🔗 Mở Spotify]({sp_result['url']})", unsafe_allow_html=True)

                lyrics = get_lyrics(sp_result['song'], sp_result['artist'])
                if lyrics:
                    emotion = clf.predict(lyrics)
                    st.markdown(f"🔮 Dự đoán cảm xúc bài hát: **{emotion.upper()}**")
                    results = recommender.recommend_from_lyrics(lyrics, target_emotion=emotion)
                else:
                    st.error("❌ Không tìm thấy lyrics để phân tích.")
            else:
                st.error("❌ Không tìm thấy bài hát trên Spotify.")

        if results is not None:
            if not matched.empty:
                top_emotion = results["emotion"].value_counts().idxmax()
                st.markdown(f"🔮 Cảm xúc đa số trong các bài tương tự: **{top_emotion.upper()}**")

            st.markdown("### 🎧 Gợi ý các bài hát tương tự:")
            for i, row in results.iterrows():
                col1, col2 = st.columns([1, 5])
                with col1:
                    sp = search_track(f"{row['song']} {row['artist']}")
                    if sp and sp['image']:
                        st.image(sp['image'], width=100)
                with col2:
                    st.markdown(f"**{row['song']}***{row['artist']}*Emotion: {row['emotion']}")
                    if sp and sp['url']:
                        st.markdown(f"[Mở trên Spotify]({sp['url']})", unsafe_allow_html=True)

elif option == "Tìm bài hát theo lời nhập tự do":
    user_input = st.text_input("✍️ Nhập một câu mô tả cảm xúc, suy nghĩ hoặc lời bài hát:")
    if user_input:
        predicted_emotion = clf.predict(user_input)
        st.success(f"💡 Dự đoán cảm xúc: **{predicted_emotion.upper()}**")

        df = recommender.df
        filtered = df[df["emotion"].str.lower() == predicted_emotion.lower()]
        if not filtered.empty:
            st.markdown("### 🎧 Các bài hát phù hợp với cảm xúc của bạn:")
            for i, row in filtered.sample(5).iterrows():
                col1, col2 = st.columns([1, 5])
                with col1:
                    sp = search_track(f"{row['song']} {row['artist']}")
                    if sp and sp['image']:
                        st.image(sp['image'], width=100)
                with col2:
                    st.markdown(f"**{row['song']}***{row['artist']}*Emotion: {row['emotion']}")
                    if sp and sp['url']:
                        st.markdown(f"[Mở trên Spotify]({sp['url']})", unsafe_allow_html=True)
        else:
            st.warning("Không tìm thấy bài hát nào phù hợp với cảm xúc này.")
