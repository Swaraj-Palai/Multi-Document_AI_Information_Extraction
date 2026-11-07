# Script to update GENERAL_LLM_PROMPT in retriver.py
# This adds capability descriptions after line 452

with open(r'c:\Users\swara\Downloads\pdf-information-extractor-main\app\componenets\retriver.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# New capability content to insert after line 452 (index 451)
caps_content = [
'=================================================\n',
'KEY CAPABILITIES\n',
'=================================================\n',
'\\n',
'When users ask about your capabilities or how to use the system, you can now describe:\n',
'\\n',
'1. Document Processing Capabilities:\n',
'   - PDF, DOCX, TXT, MD, CSV, JSON, HTML, XML, and 59+ other document formats\n',
'   - Text extraction with automatic OCR for scanned images (JPG, PNG, GIF, BMP, TIFF, WEBP)\n',
'   - Archive extraction (ZIP, RAR, 7Z, TAR, GZ) with indexed content\n',
'   - Email parsing (EML, MSG, MBOX, PST)\n',
'   - E-book reading (EPUB, MOBI, AZW3, FB2)\n',
'   - Office formats (DOC, DOCM, PPT, PPTM, XLSM, XLSB, etc.)\n',
'   - ML model metadata (H5, PT, PTH, ONNX, GGUF, TFLITE)\n',
'   - Audio/Video metadata (MP3, WAV, MP4, MKV, MOV)\n',
'   - Config and data files (YAML, YML, TOML, INI, LOG, SQL)\n',
'\\n',
'2. RAG System Features:\n',
'   - Document upload and indexing\n',
'   - Semantic search across uploaded documents\n',
'   - Question answering with document citations\n',
'   - Conversation-aware RAG (remembers context, resolves pronouns)\n',
'   - Persistent chat history in SQLite\n',
'   - Configurable chunking and retrieval parameters\n',
'\\n',
'3. System Usage:\n',
'   - Upload documents via the web interface\n',
'   - Ask questions grounded in document content\n',
'   - Receive answers with document and page citations\n',
'   - Use follow-up questions to explore topics deeper\n',
'   - Switch between RAG mode (with docs) and General mode (general knowledge)\n',
'\\n',
'=================================================\n',
'AVAILABLE DOCUMENT CATEGORIES\n',
'=================================================\n',
'\\n',
'• Documents and Text: PDF, DOCX, TXT, MD, CSV, JSON, HTML, XML\n',
'• E-books: EPUB, MOBI, AZW3, FB2\n',
'• OpenDocument: ODT, ODP, ODS\n',
'• Office: DOC, DOCM, DOTX, PPT, PPTM, PPSX, XLSM, XLSB, XLSX\n',
'• Images (OCR): JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP\n',
'• Archives: ZIP, RAR, 7Z, TAR, GZ\n',
'• Email: EML, MSG, MBOX, PST\n',
'• XPS, ML Models (metadata), Audio/Video (metadata)\n',
'\\n',
'=================================================\n',
'HOW TO USE THE SYSTEM\n',
'=================================================\n',
'\\n',
'1. Upload a document using the web interface\n',
'2. Wait for processing to complete\n',
'3. Ask questions about the document content\n',
'3. Receive answers with document and page citations\n',
'4. Use follow-up questions to explore topics deeper\n',
'5. Switch to general knowledge mode when no documents are uploaded\n',
'\\n',
'=================================================\n',
'IMPORTANT NOTES\n',
'=================================================\n',
'\\n',
'- In RAG mode: Answers are grounded in uploaded document evidence\n',
'- In General mode: Answers use general knowledge\n',
'- Always cite document sources when answering from uploaded files\n',
'- The system supports 59+ document formats for comprehensive coverage\n',
'\\n',
'=================================================\n',
END

# Insert after line 452 (which is "6. Consistency:")
# lines[452] is the line "6. Consistency:\n"
# We want to insert after this line, so at index 452 (which gives us lines[:452] + new + lines[452:])
# But actually lines[452] is the 453rd line. Let me check: if we want after line 452 (1-indexed),
# that's index 452 in 0-indexed.

insert_index = 452  # Insert after line 452 (1-indexed), which is lines[451] in 0-indexed
# Actually, let me re-examine: the user said "after line 452"
# Looking at the output, line 452 is "6. Consistency:\n"
# We want to insert AFTER this line, so at index 452 (0-indexed) would put it after line 452

new_lines = lines[:insert_index] + caps_content + lines[insert_index:]

with open(r'c:\Users\swara\Downloads\pdf-information-extractor-main\app\componenets\retriver.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Capability descriptions inserted successfully at line', insert_index + 1)