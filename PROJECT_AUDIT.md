# PROJECT AUDIT

## Repository Overview
- Framework: Flask
- RAG: LangChain + FAISS + Groq
- Embeddings: BGE-M3 (multilingual, 1024-dim) with sentence-transformers
- OCR: Indra OCR, PaddleOCR/PaddlePaddle, PyMuPDF
- DB: SQLite (chat_history.db in app/componenets/)
- Frontend: HTML/CSS/JS

## Key Files Inspected
- app/application.py - Main Flask app with routes
- app/componenets/document_loader.py - PDF/DOCX loading, OCR integration
- app/componenets/ocr_engine.py - Indra OCR wrapper
- app/componenets/embeddings.py / embeddings_bge.py - Embedding models
- app/componenets/vector_store.py - FAISS operations
- app/componenets/retriver.py - RAG query logic
- app/componenets/query_rewriter.py - Query rewriting
- app/componenets/history.py - Chat history
- app/rag/grounded_answer.py - Grounded answering
- app/structured/extractor.py - Structured extraction
- app/templates/index.html - Frontend
- app/config/config.py - Configuration

## Identified Issues
1. Security: No authentication, secret generated randomly on restart, DB in package dir
2. File upload: Basic validation, needs stronger checks
3. OCR: Good integration but needs better page classification
4. Vector store: Dimension mismatch issue fixed but needs better compatibility checks
5. Retrieval: Document filtering semantics need improvement
6. Evidence verification: Basic overlap check needs strengthening
7. Git: .env tracked? DB tracked? Needs .gitignore updates
8. README has merge conflicts
9. config/__int__.py typo (should be __init__.py)

## Working Features
- Document upload (PDF/DOCX)
- OCR with Indra
- BGE-M3 embeddings
- FAISS vector store
- Groq LLM integration
- Chat history
- Document management
