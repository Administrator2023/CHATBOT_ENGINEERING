import os

BASE_DIR = r"C:\CHATBOT_ENGINEERING"
OPENAI_API_KEY = "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"
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store.faiss")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OCR_TEMP_DIR = os.path.join(BASE_DIR, "temp_ocr")
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OCR_TEMP_DIR, exist_ok=True)
