"""
Structured document extraction with field association
"""
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field

from app.common.logger import get_logger

logger = get_logger(__name__)


class ExtractedField(BaseModel):
    field: str
    section: str = ""
    value: str = ""
    data_type: str = "string"
    page_number: int = 1
    evidence_quote: str = ""
    bounding_box: Optional[List[float]] = None
    extraction_method: str = "text"
    ocr_confidence: Optional[float] = None
    validation_status: str = "extracted"


class StructuredDocument(BaseModel):
    document_type: str = "unknown"
    document_id: str = ""
    fields: List[ExtractedField] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def extract_certificate_fields(text: str, page_num: int = 1) -> List[ExtractedField]:
    """Extract common certificate fields with proper association"""
    fields = []
    
    # Student name - look for patterns but be careful not to confuse with parents
    # Pattern: "Certified that" or "This is to certify that" followed by name
    patterns = [
        (r'Certified that\s+([A-Z][A-Z\s]+?)(?:\s+S/o|\s+D/o|\s+Son|\s+Daughter|,|\.|$)', 'student_name', 'student_details'),
        (r'Student\s*:?\s*([A-Z][A-Z\s]+)', 'student_name', 'student_details'),
        (r'Name\s+of\s+the\s+Student\s*:?\s*([A-Z][A-Z\s]+)', 'student_name', 'student_details'),
    ]
    
    for pattern, field_name, section in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            # Avoid if it contains father/mother keywords
            if not re.search(r'father|mother|parent', value, re.IGNORECASE):
                fields.append(ExtractedField(
                    field=field_name,
                    section=section,
                    value=value,
                    page_number=page_num,
                    evidence_quote=match.group(0),
                    extraction_method="regex"
                ))
                break
    
    # Father name
    patterns_father = [
        r"(?:Father'?s?\s+Name|S/o|Son of)\s*:?\s*([A-Z][A-Z\s]+?)(?:,|\.|\n|Mother)",
        r"Father\s*:?\s*([A-Z][A-Z\s]+)",
    ]
    for pattern in patterns_father:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value and not re.search(r'mother|student', value, re.IGNORECASE):
                fields.append(ExtractedField(
                    field='father_name',
                    section='parent_details',
                    value=value,
                    page_number=page_num,
                    evidence_quote=match.group(0),
                    extraction_method="regex"
                ))
                break
    
    # Mother name
    patterns_mother = [
        r"(?:Mother'?s?\s+Name|D/o|Daughter of)\s*:?\s*([A-Z][A-Z\s]+?)(?:,|\.|\n)",
        r"Mother\s*:?\s*([A-Z][A-Z\s]+)",
    ]
    for pattern in patterns_mother:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value and not re.search(r'father|student', value, re.IGNORECASE):
                fields.append(ExtractedField(
                    field='mother_name',
                    section='parent_details',
                    value=value,
                    page_number=page_num,
                    evidence_quote=match.group(0),
                    extraction_method="regex"
                ))
                break
    
    # Roll number
    roll_match = re.search(r'Roll\s*(?:No\.?|Number)\s*:?\s*([A-Z0-9]+)', text, re.IGNORECASE)
    if roll_match:
        fields.append(ExtractedField(
            field='roll_number',
            section='identification',
            value=roll_match.group(1),
            page_number=page_num,
            evidence_quote=roll_match.group(0),
            extraction_method="regex"
        ))
    
    # Date of birth
    dob_match = re.search(r'(?:Date\s+of\s+Birth|DOB)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
    if dob_match:
        fields.append(ExtractedField(
            field='date_of_birth',
            section='student_details',
            value=dob_match.group(1),
            page_number=page_num,
            evidence_quote=dob_match.group(0),
            extraction_method="regex"
        ))
    
    return fields


def extract_table_data(text: str, page_num: int = 1) -> List[Dict[str, Any]]:
    """Extract table data preserving row structure"""
    tables = []
    # Simple table extraction - look for patterns like Subject | Code | Marks
    lines = text.split('\n')
    # Look for lines with multiple tabular elements
    table_rows = []
    for line in lines:
        # Check if line looks like a table row (has numbers and text)
        if re.search(r'\d+', line) and len(line.split()) >= 2:
            # Avoid headers that are just labels
            if not re.match(r'^\s*(Page|Date|Serial|Certificate)\s*:', line, re.IGNORECASE):
                table_rows.append(line.strip())
    
    if table_rows:
        tables.append({
            "page": page_num,
            "rows": table_rows,
            "type": "marks_table"
        })
    
    return tables
