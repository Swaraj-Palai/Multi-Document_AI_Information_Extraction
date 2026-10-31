import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1000"))

# Query Rewriting Configuration (First LLM Stage - before Embedding Model)
ENABLE_QUERY_REWRITING = os.environ.get("ENABLE_QUERY_REWRITING", "true").lower() == "true"
QUERY_REWRITE_MODEL = os.environ.get("QUERY_REWRITE_MODEL", LLM_MODEL)
QUERY_REWRITE_TEMPERATURE = float(os.environ.get("QUERY_REWRITE_TEMPERATURE", "0.3"))
QUERY_REWRITE_MAX_TOKENS = int(os.environ.get("QUERY_REWRITE_MAX_TOKENS", "200"))

# Embeddings & Vector Store Configuration
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "bge-m3").lower()  # bge-m3 or sentence-transformers
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3") if EMBEDDING_PROVIDER == "bge-m3" else os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DB_FAISS_PATH = os.environ.get("VECTOR_DB_PATH", os.environ.get("DB_FAISS_PATH", "vector_db/db_faiss"))
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
DATA_PATH = os.environ.get("DATA_PATH", "data/")

# BGE-M3 Configuration
BGE_M3_MODEL_PATH = os.environ.get("BGE_M3_MODEL_PATH", os.path.join(os.getcwd(), "models", "bge-m3"))
BGE_M3_DEVICE = os.environ.get("BGE_M3_DEVICE", "cpu")
BGE_M3_BATCH_SIZE = int(os.environ.get("BGE_M3_BATCH_SIZE", "1"))
BGE_M3_MAX_LENGTH = int(os.environ.get("BGE_M3_MAX_LENGTH", "1024"))

# Chunking & Retrieval Parameters
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
TOP_K = int(os.environ.get("TOP_K", "4"))

# Upload Constraints
ALLOWED_EXTENSIONS = {"pdf", "docx"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB limit

# OCR Configuration (Indra OCR / PaddleOCR backend)
OCR_ENABLED = os.environ.get("OCR_ENABLED", "true").lower() == "true"
OCR_THREADS = int(os.environ.get("OCR_THREADS", "4"))
OCR_DPI = int(os.environ.get("OCR_DPI", "250"))
OCR_MODE = os.environ.get("OCR_MODE", "auto")  # auto | text | ocr
OCR_TEXT_THRESHOLD = int(os.environ.get("OCR_TEXT_THRESHOLD", "50"))
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")  # Path to tesseract if needed
