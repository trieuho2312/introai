import re

def clean_text(text):
    """Làm sạch văn bản: loại bỏ ký tự đặc biệt, chữ thường hóa, bỏ từ ngắn."""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    words = [word for word in text.split() if len(word) > 2]
    return " ".join(words)