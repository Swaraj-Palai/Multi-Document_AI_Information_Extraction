"""
Security utilities for file validation, hashing, and sanitization
"""
import os
import hashlib
try:
    import magic
except ImportError:
    magic = None
from pathlib import Path
from typing import Tuple, Optional

ALLOWED_MIME_TYPES = {
    # PDF & Word
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/msword': '.doc',
    'application/vnd.ms-word.document.macroEnabled.12': '.docm',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.template': '.dotx',
    # Plain text / markup / data
    'text/plain': '.txt',
    'text/markdown': '.md',
    'text/rtf': '.rtf',
    'application/rtf': '.rtf',
    'text/csv': '.csv',
    'text/tab-separated-values': '.tsv',
    'application/json': '.json',
    'application/xml': '.xml',
    'text/xml': '.xml',
    'text/html': '.html',
    'application/yaml': '.yaml',
    'text/yaml': '.yaml',
    'application/toml': '.toml',
    'text/x-toml': '.toml',
    'text/ini': '.ini',
    'text/plain': '.log',
    'application/sql': '.sql',
    'text/sql': '.sql',
    # E-books
    'application/epub+zip': '.epub',
    'application/x-mobipocket-ebook': '.mobi',
    'application/vnd.amazon.ebook': '.azw3',
    'application/x-fictionbook+xml': '.fb2',
    # OpenDocument
    'application/vnd.oasis.opendocument.text': '.odt',
    'application/vnd.oasis.opendocument.presentation': '.odp',
    'application/vnd.oasis.opendocument.spreadsheet': '.ods',
    # Office (PowerPoint)
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/vnd.ms-powerpoint': '.ppt',
    'application/vnd.ms-powerpoint.presentation.macroEnabled.12': '.pptm',
    'application/vnd.openxmlformats-officedocument.presentationml.slideshow': '.ppsx',
    # Office (Excel)
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/vnd.ms-excel': '.xls',
    'application/vnd.ms-excel.sheet.macroEnabled.12': '.xlsm',
    'application/vnd.ms-excel.sheet.binary.macroEnabled.12': '.xlsb',
    # Images
    'image/jpeg': '.jpg',
    'image/jpeg': '.jpeg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
    'image/tiff': '.tif',
    'image/webp': '.webp',
    # Archives
    'application/zip': '.zip',
    'application/vnd.rar': '.rar',
    'application/x-7z-compressed': '.7z',
    'application/x-tar': '.tar',
    'application/gzip': '.gz',
    # Email
    'message/rfc822': '.eml',
    'application/vnd.ms-outlook': '.msg',
    'application/mbox': '.mbox',
    'application/vnd.ms-outlook': '.pst',
    # XPS
    'application/vnd.ms-xpsdocument': '.xps',
    # ML Models
    'application/x-hdf5': '.h5',
    'application/x-torch': '.pt',
    'application/x-torch': '.pth',
    'application/x-onnx': '.onnx',
    'application/x-gguf': '.gguf',
    'application/x-tflite': '.tflite',
}

PDF_MAGIC = b'%PDF'
ZIP_MAGIC = b'PK'
OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
JPEG_MAGIC = b'\xff\xd8\xff'
GIF_MAGIC = b'GIF8'
BMP_MAGIC = b'BM'
TIFF_MAGIC_LE = b'II*\x00'
TIFF_MAGIC_BE = b'MM\x00*'
WEBP_MAGIC = b'RIFF'
RAR_MAGIC = b'Rar!\x1a\x07\x00'
GZ_MAGIC = b'\x1f\x8b'
SEVEN_Z_MAGIC = b'7z\xbc\xaf\x27\x1c'
H5_MAGIC = b'\x89HDF\r\n\x1a\n'
ONNX_MAGIC = b'\x08\x09\x10\x13'


def validate_file_signature(file_path: str) -> Tuple[bool, str]:
    """Validate file magic bytes for known types; permissive for text/data files."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)

        ext = Path(file_path).suffix.lower()

        # ---- PDF ----
        if ext == '.pdf':
            if header.startswith(PDF_MAGIC):
                return True, 'pdf'
            return False, 'Invalid PDF signature'

        # ---- ZIP-based office / e-book / archive formats ----
        if ext in ('.docx', '.docm', '.dotx', '.pptx', '.pptm', '.ppsx',
                   '.xlsx', '.xlsm', '.xlsb', '.epub', '.odt', '.odp', '.ods', '.jar'):
            if header.startswith(ZIP_MAGIC):
                return True, ext.lstrip('.')
            return False, 'Invalid ' + ext.lstrip('.') + ' signature'

        # ---- OLE2 compound document (legacy Office / Outlook) ----
        if ext in ('.doc', '.xls', '.ppt', '.msg', '.pst'):
            if header.startswith(OLE_MAGIC):
                return True, ext.lstrip('.')
            return False, 'Invalid ' + ext.lstrip('.') + ' signature'

        # ---- Images ----
        if ext in ('.jpg', '.jpeg'):
            if header.startswith(JPEG_MAGIC):
                return True, ext.lstrip('.')
            return False, 'Invalid JPEG signature'
        if ext == '.png':
            if header.startswith(PNG_MAGIC):
                return True, 'png'
            return False, 'Invalid PNG signature'
        if ext in ('.gif',):
            if header.startswith(GIF_MAGIC):
                return True, 'gif'
            return False, 'Invalid GIF signature'
        if ext == '.bmp':
            if header.startswith(BMP_MAGIC):
                return True, 'bmp'
            return False, 'Invalid BMP signature'
        if ext in ('.tiff', '.tif'):
            if header.startswith(TIFF_MAGIC_LE) or header.startswith(TIFF_MAGIC_BE):
                return True, 'tiff'
            return False, 'Invalid TIFF signature'
        if ext == '.webp':
            if header.startswith(WEBP_MAGIC) and header[8:12] == b'WEBP':
                return True, 'webp'
            return False, 'Invalid WEBP signature'

        # ---- Archives ----
        if ext == '.rar':
            if header.startswith(RAR_MAGIC):
                return True, 'rar'
            return False, 'Invalid RAR signature'
        if ext == '.gz':
            if header.startswith(GZ_MAGIC):
                return True, 'gz'
            return False, 'Invalid GZ signature'
        if ext == '.7z':
            if header.startswith(SEVEN_Z_MAGIC):
                return True, '7z'
            return False, 'Invalid 7Z signature'

        # ---- ML Models ----
        if ext == '.h5':
            if header.startswith(H5_MAGIC):
                return True, 'h5'
            return False, 'Invalid H5 signature'
        if ext in ('.pt', '.pth'):
            if header.startswith(ZIP_MAGIC):
                return True, ext.lstrip('.')
            return False, 'Invalid ' + ext.lstrip('.') + ' signature'
        if ext == '.onnx':
            if header.startswith(ONNX_MAGIC):
                return True, 'onnx'
            return False, 'Invalid ONNX signature'

        # ---- XPS ----
        if ext == '.xps':
            if header.startswith(ZIP_MAGIC):
                return True, 'xps'
            return False, 'Invalid XPS signature'

        # ---- Text / config / data formats: accept by extension ----
        if ext in ('.txt', '.md', '.rtf', '.csv', '.tsv', '.json', '.xml', '.html', '.htm',
                   '.yaml', '.yml', '.toml', '.ini', '.log', '.sql'):
            return True, ext.lstrip('.')

        # ---- Audio / Video: permissive (speech-to-text may not be available) ----
        if ext in ('.mp3', '.wav', '.mp4', '.mkv', '.mov'):
            return True, ext.lstrip('.')

        # ---- Unknown extension: try UTF-8 decode ----
        try:
            header.decode('utf-8')
            return True, ext.lstrip('.')
        except Exception:
            return False, 'Unsupported file type'

    except Exception as e:
        return False, str(e)


def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def secure_filename_custom(filename: str) -> str:
    from werkzeug.utils import secure_filename
    import uuid
    name = secure_filename(filename)
    if not name:
        name = 'upload_' + uuid.uuid4().hex[:8]
    ext = Path(name).suffix
    return uuid.uuid4().hex + ext
