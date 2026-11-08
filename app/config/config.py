import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1000"))

# Query Rewriting Configuration (First LLM Stage - before Embedding Model)
ENABLE_QUERY_REWRITING = os.environ.get("ENABLE_QUERY_REWRITING", "true").lower() == "true"
QUERY_REWRITE_MODEL = os.environ.get("QUERY_REWRITE_MODEL", LLM_MODEL)
QUERY_REWRITE_TEMPERATURE = float(os.environ.get("QUERY_REWRITE_TEMPERATURE", "0.7"))
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
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "500"))
TOP_K = int(os.environ.get("TOP_K", "8"))

# ---------------------------------------------------------------------------
# Query Understanding Module (Pre-Retrieval Stage)
# Analyzes the user's question BEFORE document retrieval and produces
# structured retrieval instructions. Does NOT generate the final answer.
# ---------------------------------------------------------------------------
QUERY_UNDERSTANDING_ENABLED = os.environ.get("QUERY_UNDERSTANDING_ENABLED", "true").lower() == "true"
QUERY_UNDERSTANDING_USE_LLM = os.environ.get("QUERY_UNDERSTANDING_USE_LLM", "false").lower() == "true"
QUERY_UNDERSTANDING_DEFAULT_TOP_K = int(os.environ.get("QUERY_UNDERSTANDING_DEFAULT_TOP_K", str(TOP_K)))
QUERY_UNDERSTANDING_COMPLEX_BOOST_TOP_K = int(os.environ.get("QUERY_UNDERSTANDING_COMPLEX_BOOST_TOP_K", "8"))

# ---------------------------------------------------------------------------
# General LLM (normal chat) mode
# Lets the app behave like a normal LLM when no documents are uploaded,
# while keeping all RAG/document-grounded behaviour unchanged when documents exist.
# ---------------------------------------------------------------------------
ENABLE_GENERAL_LLM = os.environ.get("ENABLE_GENERAL_LLM", "true").lower() == "true"
GENERAL_LLM_TEMPERATURE = float(os.environ.get("GENERAL_LLM_TEMPERATURE", "0.7"))
GENERAL_LLM_MAX_TOKENS = int(os.environ.get("GENERAL_LLM_MAX_TOKENS", "1000"))
# When documents exist but nothing relevant is found, fall back to general LLM?
# Default False to preserve the strict grounded "not found" behaviour.
GENERAL_FALLBACK_WHEN_NO_MATCH = os.environ.get("GENERAL_FALLBACK_WHEN_NO_MATCH", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Conversation-aware RAG
# Remembers previous messages within the same chat so the app understands
# follow-up questions and pronouns (it, this, that, they) without the user
# repeating everything. Previous messages help understand the QUESTION;
# uploaded documents provide the EVIDENCE for answers.
# ---------------------------------------------------------------------------
CONVERSATION_AWARE_ENABLED = os.environ.get("CONVERSATION_AWARE_ENABLED", "true").lower() == "true"
CONVERSATION_AWARE_MAX_MESSAGES = int(os.environ.get("CONVERSATION_AWARE_MAX_MESSAGES", "10"))
CONVERSATION_AWARE_MAX_TOKENS = int(os.environ.get("CONVERSATION_AWARE_MAX_TOKENS", "2000"))
CONVERSATION_AWARE_USE_LLM = os.environ.get("CONVERSATION_AWARE_USE_LLM", "false").lower() == "true"
CONVERSATION_AWARE_NEW_TOPIC_THRESHOLD = float(os.environ.get("CONVERSATION_AWARE_NEW_TOPIC_THRESHOLD", "0.5"))

# Upload Constraints
# Universal document support: text, office, images, archives, email, e-books,
# audio/video, ML models, and more.
# Add or remove extensions here to control what the app accepts.
ALLOWED_EXTENSIONS = {
    # ---- PDF & Word ----
    "pdf", "docx",
    # ---- Plain text / markup / data ----
    "txt", "md", "rtf",
    "csv", "tsv",
    "json", "xml",
    "html", "htm",
    "yaml", "yml",
    "toml",
    "ini",
    "log",
    "sql",
    # ---- E-books ----
    "epub",
    "mobi", "azw3", "fb2",
    # ---- OpenDocument ----
    "odt", "odp", "ods",
    # ---- Office (Word) ----
    "doc", "docm", "dotx",
    # ---- Office (PowerPoint) ----
    "pptx", "ppt", "pptm", "ppsx",
    # ---- Office (Excel) ----
    "xlsx", "xls", "xlsm", "xlsb",
    # ---- Images (OCR) ----
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp",
    # ---- Archives ----
    "zip", "rar", "7z", "tar", "gz",
    # ---- Email ----
    "eml", "msg", "mbox", "pst",
    # ---- XPS ----
    "xps",
    # ---- ML Models (binary metadata) ----
    "h5", "pt", "pth", "onnx", "gguf", "tflite",
}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB limit for broader file types

# OCR Configuration (Indra OCR / PaddleOCR backend)
OCR_ENABLED = os.environ.get("OCR_ENABLED", "true").lower() == "true"
OCR_BACKEND = os.environ.get("OCR_BACKEND", "indra").lower()  # indra | tesseract
OCR_LANG = os.environ.get("OCR_LANG", "eng")
OCR_THREADS = int(os.environ.get("OCR_THREADS", "4"))
OCR_DPI = int(os.environ.get("OCR_DPI", "250"))
OCR_MODE = os.environ.get("OCR_MODE", "auto")  # auto | text | ocr
OCR_TEXT_THRESHOLD = int(os.environ.get("OCR_TEXT_THRESHOLD", "50"))
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")  # Path to tesseract if needed
