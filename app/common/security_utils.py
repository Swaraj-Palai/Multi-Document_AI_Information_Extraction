"""
Security utilities for file validation, hashing, and sanitization
"""
import os
import hashlib
import magic
from pathlib import Path
from typing import Tuple, Optional

ALLOWED_MIME_TYPES = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
}

PDF_MAGIC = b'%PDF'
ZIP_MAGIC = b'PK'  # docx is a zip


def validate_file_signature(file_path: str) -> Tuple[bool, str]:
    """Validate file magic bytes"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
        
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            if header.startswith(PDF_MAGIC):
                return True, 'pdf'
            return False, 'Invalid PDF signature'
        elif ext == '.docx':
            if header.startswith(ZIP_MAGIC):
                return True, 'docx'
            return False, 'Invalid DOCX signature'
        return False, 'Unsupported file type'
    except Exception as e:
        return False, str(e)


def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def secure_filename_custom(filename: str) -> str:
    """Generate secure filename"""
    from werkzeug.utils import secure_filename
    import uuid
    name = secure_filename(filename)
    if not name:
        name = f"upload_{uuid.uuid4().hex[:8]}"
    ext = Path(name).suffix
    return f"{uuid.uuid4().hex}{ext}"
