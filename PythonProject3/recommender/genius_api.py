import os
from dotenv import load_dotenv
import lyricsgenius

load_dotenv()
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

if not GENIUS_ACCESS_TOKEN:
    raise ValueError("❌ GENIUS_ACCESS_TOKEN chưa được thiết lập trong file .env")

# Khởi tạo Genius client
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=10, retries=3)
genius.skip_non_songs = True
genius.excluded_terms = ["(Remix)", "(Live)"]
genius.verbose = False  # Không in log ra màn hình

def get_lyrics(song_title: str, artist_name: str) -> str | None:
    """
    Trả về lời bài hát từ Genius nếu tìm được. Nếu không, trả về None.
    """
    try:
        song = genius.search_song(title=song_title, artist=artist_name)
        if song and song.lyrics:
            return song.lyrics
        return None
    except Exception as e:
        print(f"❌ Lỗi khi gọi Genius API: {e}")
        return None
