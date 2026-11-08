# Universal Multi-Document AI Information Extraction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-red.svg)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/RAG-LangChain-green.svg)](https://python.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-purple.svg)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)

A Flask-based, multi-document Retrieval-Augmented Generation (RAG) application. Upload supported documents, extract and split their content, index the resulting text chunks in FAISS, and ask questions using document-grounded retrieval and source/page references. The application also includes OCR integration, query understanding and rewriting, conversation-aware follow-ups, persistent chat history, and a general-chat mode when no document index is available.

> **Embedding model:** The default configuration uses **BAAI BGE-M3** for multilingual dense text embeddings. The project also supports a configurable Hugging Face Sentence Transformers model and has a multilingual MiniLM fallback if embedding initialization fails.

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Embedding model](#embedding-model)
- [Supported file types and extraction notes](#supported-file-types-and-extraction-notes)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [How document retrieval works](#how-document-retrieval-works)
- [API overview](#api-overview)
- [Important implementation notes](#important-implementation-notes)

## Features

- **Multi-document ingestion:** Upload supported document formats through the web interface.
- **Text extraction:** Uses format-specific loaders and extraction fallbacks. PDF processing checks for a usable text layer and can invoke Indra OCR for scanned or image-based pages.
- **OCR integration:** Indra OCR with a PaddleOCR backend is configurable through environment settings.
- **Text chunking:** Uses LangChain's Recursive Character Text Splitter and preserves source, page, file type, document ID, and chunk metadata where available.
- **Multilingual embeddings:** BAAI BGE-M3 is the default embedding model. Embedding instances are cached to avoid repeatedly loading the model.
- **Persistent vector index:** FAISS stores document vectors locally and supports incremental document indexing.
- **Query understanding:** A pre-retrieval component analyses the user's question and can suggest retrieval settings.
- **Query rewriting:** An optional LLM stage can rewrite or expand a query before retrieval.
- **Conversation-aware RAG:** The application can use earlier messages in the same chat to resolve follow-up questions and references. Conversation history is bounded by configuration.
- **Grounded answers:** When documents are indexed, retrieved document passages are used as the evidence for the answer. The application can return a not-found response when relevant evidence is unavailable.
- **General-chat mode:** When no vector store is available, the application can answer general questions if general LLM mode is enabled.
- **Chat history and document management:** Stores conversations in SQLite and maintains metadata for uploaded documents.
- **Web interface:** Flask-rendered interface with browser-side JavaScript for chat and document interactions.

## Architecture

### Document ingestion and indexing

~~~text
Document upload
      |
      v
Format-aware document loader
      |
      +---- Digital PDF / office / text content
      |
      +---- Scanned PDF pages --> Indra OCR (when available)
      |
      v
Text cleaning and metadata assignment
      |
      v
Recursive character text splitting
      |
      v
BAAI BGE-M3 embedding model (default)
      |
      v
FAISS vector index
      |
      v
Persist index and document metadata
~~~

### Question answering

~~~text
User question
      |
      v
Conversation context and query understanding
      |
      v
Optional query rewriting
      |
      v
Embedding + FAISS similarity search
      |
      v
Validate retrieved chunks and document metadata
      |
      v
Build source/page-aware context
      |
      v
Groq-hosted LLM and grounding instructions
      |
      v
Answer + retrieved source references
~~~

## Embedding model

### Default: BAAI BGE-M3

- **Model ID:** [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Provider setting:** EMBEDDING_PROVIDER=bge-m3
- **Model setting:** EMBEDDING_MODEL=BAAI/bge-m3
- **Embedding implementation:** Sentence Transformers through a LangChain-compatible wrapper.
- **Default device:** CPU.
- **Default batch size:** 1.
- **Vector type used by this project:** Dense embeddings.
- **Embedding dimension:** 1024 for the BGE-M3 dense representation.
- **Normalization:** Enabled for document and query embeddings.

The implementation calls Sentence Transformers to encode document text and normalizes the resulting vectors. Query embeddings use the same embedding wrapper. FAISS then performs similarity search against the stored vectors.

BGE-M3 supports dense, sparse, and multi-vector representations as a model family. **This repository currently uses the dense embedding path; it does not implement BGE-M3's sparse or multi-vector retrieval modes.**

### Alternative and fallback model

The generic embedding loader can also use a Hugging Face Sentence Transformers model. The configured non-BGE default and initialization fallback are:

~~~text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
~~~

To select that model explicitly, set:

~~~env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
~~~

If initialization of the configured embedding model fails, the current embedding loader attempts to initialize the multilingual MiniLM model as a fallback. Check application logs to confirm which model initialized successfully in a particular run.

### Local model download (optional)

The application can load BGE-M3 from the configured local model directory when that directory contains the model configuration. Otherwise, it loads the model ID from Hugging Face.

To download the model locally:

~~~bash
python scripts/download_bge_m3.py
~~~

The default local directory is models/bge-m3. It can be changed with BGE_M3_MODEL_PATH.

> **Important:** Use the same embedding model and compatible embedding dimensions for every vector in a FAISS index. If you change embedding models, rebuild the index and re-embed the documents rather than mixing vectors from different models.

## Supported file types and extraction notes

The allowed extensions are defined in the application's configuration. They include:

| Category | Extensions |
|---|---|
| PDF and Word | PDF, DOCX, DOC, DOCM, DOTX |
| Text and markup | TXT, MD, RTF, HTML, HTM, XML, YAML, YML, TOML, INI, LOG, SQL |
| Structured data | CSV, TSV, JSON |
| PowerPoint | PPTX, PPT, PPTM, PPSX |
| Spreadsheets | XLSX, XLS, XLSM, XLSB |
| OpenDocument | ODT, ODP, ODS |
| E-books | EPUB, MOBI, AZW3, FB2 |
| Images | JPG, JPEG, PNG, GIF, BMP, TIFF, TIF, WEBP |
| Archives | ZIP, RAR, 7Z, TAR, GZ |
| Email | EML, MSG, MBOX, PST |
| Other | XPS, H5, PT, PTH, ONNX, GGUF, TFLITE |

**Extraction depends on the format.** Text-bearing files are processed using format-specific loaders or fallback loaders. Scanned PDFs can use Indra OCR when the OCR engine and its dependencies are available. In the current implementation, audio/video files are recorded as metadata placeholders; automatic speech transcription is not configured. Binary machine-learning model files are also recorded as metadata rather than having their internal model contents extracted. Archive extraction is format-dependent and may require optional system or Python dependencies.

The configured maximum request content size is 100 MB by default. Actual successful extraction also depends on the installed dependencies and the content of the uploaded file.

## Tech stack

| Area | Technologies |
|---|---|
| Backend | Python, Flask |
| RAG orchestration | LangChain, LangChain Community, LangChain Core |
| Embeddings | BAAI BGE-M3, Sentence Transformers, LangChain Hugging Face |
| Vector store | FAISS CPU |
| LLM | Groq API through LangChain Groq |
| PDF and document processing | PyMuPDF, pypdf, python-docx, docx2txt, Unstructured and format-specific loaders |
| OCR | Indra OCR, PaddleOCR, PaddlePaddle; Tesseract dependencies are also included |
| Chat persistence | SQLite |
| Frontend | HTML, CSS, JavaScript |

## Project structure

~~~text
.
├── app/
│   ├── application.py             # Flask application and HTTP routes
│   ├── common/                    # Logging, exceptions, security helpers
│   ├── componenets/
│   │   ├── document_loader.py     # File extraction, cleaning and chunking
│   │   ├── embeddings.py          # Embedding provider selection and fallback
│   │   ├── embeddings_bge.py      # BGE-M3 embedding implementation
│   │   ├── ocr_engine.py          # Indra OCR integration
│   │   ├── query_rewriter.py      # Query rewriting
│   │   ├── query_understanding.py # Pre-retrieval query analysis
│   │   ├── conversation_context.py # Conversation context handling
│   │   ├── retriver.py             # Retrieval and RAG answer flow
│   │   ├── vector_store.py        # FAISS index and document metadata
│   │   └── history.py             # SQLite chat history
│   ├── config/
│   │   └── config.py              # Environment-based settings
│   ├── rag/
│   │   └── grounded_answer.py     # Grounding and answer support
│   ├── structured/
│   │   └── extractor.py          # Structured extraction utilities
│   ├── static/                    # Static frontend assets
│   └── templates/
│       └── index.html             # Main web interface
├── scripts/
│   └── download_bge_m3.py         # Optional local BGE-M3 download
├── req.txt                        # Python dependencies
├── run.py                         # Application entry point
└── .env.example                   # Example environment configuration
~~~

## Getting started

### 1. Clone the repository

~~~bash
git clone https://github.com/Swaraj-Palai/Multi-Document_AI_Information_Extraction.git
cd Multi-Document_AI_Information_Extraction
~~~

### 2. Create and activate a virtual environment

~~~bash
python -m venv .venv
~~~

Windows:

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

Linux / macOS:

~~~bash
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
python -m pip install --upgrade pip
pip install -r req.txt
pip install -e .
~~~

The requirements file includes OCR dependencies. If PaddlePaddle installation fails on your platform, use the CPU installation command appropriate for your Python and operating-system combination from the [PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/en/install/quick).

### 4. Configure environment variables

Copy the example environment file:

Windows:

~~~powershell
Copy-Item .env.example .env
~~~

Linux / macOS:

~~~bash
cp .env.example .env
~~~

At minimum, add your Groq API key to the .env file:

~~~env
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-120b
~~~

The repository's .env.example already includes the BGE-M3 provider, model ID, CPU settings, vector-store path, chunking, retrieval, query rewriting, conversation, OCR, and server settings.

### 5. (Optional) Download BGE-M3 before startup

~~~bash
python scripts/download_bge_m3.py
~~~

This saves the model under the configured BGE_M3_MODEL_PATH. Alternatively, the embedding loader can download/load BAAI/bge-m3 from Hugging Face when no valid local model directory is found.

### 6. Run the application

~~~bash
python run.py
~~~

The default local address is:

http://localhost:5000

The host and port can be changed with the HOST and PORT environment variables.

## Configuration

The main configuration is read from environment variables in app/config/config.py. The following settings are especially relevant:

| Variable | Example / default | Purpose |
|---|---|---|
| EMBEDDING_PROVIDER | bge-m3 | Selects BGE-M3 or the Hugging Face embedding route |
| EMBEDDING_MODEL | BAAI/bge-m3 | Embedding model ID |
| BGE_M3_MODEL_PATH | ./models/bge-m3 | Optional local BGE-M3 model directory |
| BGE_M3_DEVICE | cpu | Device used to load BGE-M3 |
| BGE_M3_BATCH_SIZE | 1 | BGE-M3 encoding batch size |
| VECTOR_DB_PATH | vector_db/db_faiss | Persistent FAISS index directory |
| CHUNK_SIZE | 1000 | Text chunk size |
| CHUNK_OVERLAP | 200 in .env.example | Overlap between chunks |
| TOP_K | 4 in .env.example | Number of relevant chunks requested by default in the example configuration |
| ENABLE_QUERY_REWRITING | true | Enables the optional query-rewriting stage |
| ENABLE_GENERAL_LLM | true | Allows general chat when no vector store is available |
| GENERAL_FALLBACK_WHEN_NO_MATCH | false | Keeps document answers grounded when no relevant chunks are found |
| CONVERSATION_AWARE_ENABLED | true | Enables conversation-aware follow-up handling |
| OCR_ENABLED | true | Enables the OCR integration |
| OCR_MODE | auto | Indra OCR mode |
| PORT | 5000 | Web server port |
| HOST | 0.0.0.0 | Web server host |

Values in .env override the fallback values in the Python configuration. See .env.example for the complete set of supported settings.

## How document retrieval works

1. A user uploads a document using the web interface.
2. The loader extracts text and attaches document metadata such as filename, page, file type, and document ID.
3. Extracted content is cleaned and divided into overlapping chunks.
4. The configured embedding model converts each chunk into a vector.
5. The vectors and associated LangChain document records are stored in the local FAISS index.
6. For a question, the application prepares conversation context and runs query understanding. Query rewriting may run when enabled and appropriate.
7. FAISS similarity search retrieves candidate chunks. The application checks retrieved metadata against the active document registry and any selected-document filter.
8. The retrieved passages are assembled into context for the grounded-answer stage and sent to the configured Groq LLM.
9. The response includes source information derived from retrieved document metadata.

When no vector store is available, general-chat behavior is controlled by ENABLE_GENERAL_LLM. When documents exist but no relevant chunks are found, GENERAL_FALLBACK_WHEN_NO_MATCH is false by default in the example configuration, preserving the document-grounded not-found behavior.

## API overview

The Flask application exposes routes for the web interface and document/chat operations. The primary routes include:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | Render the main interface |
| POST | /upload | Upload, extract, chunk and index a document |
| POST | /chat | Submit a chat question |
| GET | /documents | List uploaded document metadata |
| POST | /reset-documents | Reset the FAISS vector store and document registry |

For exact request fields and response structures, refer to the route handlers in app/application.py.

## Important implementation notes

- **Model and index compatibility:** All vectors in a FAISS index must come from the same embedding model and compatible dimensions. Rebuild/re-embed the index after changing embedding models.
- **Runtime model confirmation:** The configured default is BGE-M3, but environment variables can override it. If BGE-M3 initialization fails, the current loader attempts the multilingual MiniLM fallback. Review application logs to verify the model that actually loaded.
- **Dense retrieval only:** The current BGE-M3 integration uses dense vectors with FAISS. Sparse and multi-vector BGE-M3 retrieval are not wired into this implementation.
- **Maximum sequence length setting:** BGE_M3_MAX_LENGTH is present in the configuration, but the current Sentence Transformers loader does not explicitly assign it to the loaded model's max_seq_length. Do not assume that changing this environment variable currently changes the active model limit.
- **OCR availability:** Scanned-PDF OCR depends on Indra OCR and its runtime dependencies being installed and initialized successfully.
- **File-format behavior:** An accepted extension does not guarantee that every file's internal content can be extracted. Some binary and media formats are metadata-only in the current implementation.

---

## License

No license file was present in the repository at the time this README was updated. Check with the repository owner before reusing or distributing this project.


## Free public deployment (Oracle Cloud)

A Docker-based deployment profile is prepared for Oracle Cloud Always Free Ampere A1 ARM64. It uses BGE-M3, FAISS, Groq, per-browser isolated storage and Tesseract OCR (instead of Indra/PaddleOCR on ARM64). See the [Oracle Cloud deployment guide](deploy/oracle/README.md) on the `deploy/oracle-always-free` branch.
