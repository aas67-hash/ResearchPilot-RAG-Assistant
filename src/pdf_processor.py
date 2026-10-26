"""PDF extraction and text processing module using PyMuPDF.

This module handles:
- Validating PDF files (file type, existence, size constraints)
- Extracting page-level text and metadata from PDF research papers
- Cleaning and normalizing text (whitespace, hyphenated line breaks)
- Detecting low-text/scanned pages and flagging them for OCR
- Raising clear custom exceptions for corrupt, invalid, or empty files
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

import pymupdf


class PDFProcessingError(Exception):
    """Base exception for all PDF processing errors in ResearchPilot."""


class PDFNotFoundError(PDFProcessingError, FileNotFoundError):
    """Raised when the specified PDF file cannot be found."""


class InvalidPDFError(PDFProcessingError):
    """Raised when the file is not a valid PDF or has an invalid structure."""


class CorruptPDFError(PDFProcessingError):
    """Raised when a PDF file is corrupt, damaged, or unreadable."""


class EmptyPDFError(PDFProcessingError):
    """Raised when a PDF file is 0 bytes or contains 0 pages."""


class PDFSizeLimitError(PDFProcessingError):
    """Raised when a PDF file exceeds the permitted maximum file size."""


# Default threshold in characters to flag a page as possibly scanned / image-only
DEFAULT_MIN_CHARS_THRESHOLD: int = 50
# Default maximum allowed PDF size (in Megabytes)
DEFAULT_MAX_FILE_SIZE_MB: float = 100.0
# Warning message for low-text or scanned pages
SCANNED_PAGE_WARNING: str = "possibly scanned, OCR needed"


def validate_pdf_file(
    file_path: Union[str, Path],
    max_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
) -> Path:
    """Validate that the file exists, has a .pdf extension, is non-empty, and is within size limits.

    Args:
        file_path: Path to the target PDF file.
        max_size_mb: Maximum allowed file size in megabytes.

    Returns:
        Path: Resolved Path object of the validated file.

    Raises:
        PDFNotFoundError: If the file does not exist.
        InvalidPDFError: If the path is a directory or does not have a .pdf extension.
        EmptyPDFError: If the file size is 0 bytes.
        PDFSizeLimitError: If the file size exceeds max_size_mb.
    """
    path = Path(file_path)

    if max_size_mb <= 0:
        raise ValueError(f"max_size_mb must be positive, got {max_size_mb}")

    if not path.exists():
        raise PDFNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise InvalidPDFError(f"Target path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise InvalidPDFError(f"Invalid file extension '{path.suffix}'. Expected '.pdf'.")

    file_size_bytes = path.stat().st_size
    if file_size_bytes == 0:
        raise EmptyPDFError(f"PDF file is empty (0 bytes): {path}")

    max_size_bytes = int(max_size_mb * 1024 * 1024)
    if file_size_bytes > max_size_bytes:
        size_mb = file_size_bytes / (1024 * 1024)
        raise PDFSizeLimitError(
            f"PDF file size ({size_mb:.2f} MB) exceeds maximum allowed size ({max_size_mb:.2f} MB)."
        )

    return path


def clean_text(text: str) -> str:
    """Clean and normalize extracted PDF text.

    Transforms performed:
    - Normalizes carriage returns and unicode whitespace
    - Recombines hyphenated words split across line breaks (e.g. 'multi-\\nline' -> 'multiline')
    - Collapses multiple horizontal whitespace characters into single spaces
    - Collapses excessive consecutive newlines to double newlines (preserving paragraphs)
    - Strips leading and trailing whitespace per line and overall

    Args:
        text: Raw text string extracted from a PDF page.

    Returns:
        Cleaned and normalized text string.
    """
    if not text:
        return ""

    # Normalize line breaks and unicode non-breaking spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")

    # Rejoin words broken across line breaks with hyphens
    # Matches a word ending with hyphen at line end, followed by next line beginning
    text = re.sub(r"([a-zA-Z0-9]+)-\s*\n\s*([a-zA-Z0-9]+)", r"\1\2", text)

    # Collapse horizontal spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3 or more consecutive newlines into 2 (clean paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Clean leading/trailing spaces for each line while preserving line breaks
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned


def extract_text_by_page(
    file_path: Union[str, Path],
    min_chars_threshold: int = DEFAULT_MIN_CHARS_THRESHOLD,
    max_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
) -> List[Dict[str, Any]]:
    """Extract page-level text and metadata from a PDF research paper.

    Args:
        file_path: Path to the target PDF document.
        min_chars_threshold: Minimum character count below which a page is flagged as scanned/OCR needed.
        max_size_mb: Maximum allowed PDF file size in MB.

    Returns:
        List of dictionaries with keys:
            - 'source': Source filename (e.g. 'paper.pdf')
            - 'page_number': 1-indexed page number
            - 'text': Cleaned, normalized text content
            - 'warning': 'possibly scanned, OCR needed' if text length < min_chars_threshold, else None
            - 'is_scanned': True if flagged, False otherwise

    Raises:
        PDFNotFoundError: If the file does not exist.
        InvalidPDFError: If file has invalid extension or encrypted/unreadable structure.
        EmptyPDFError: If file is 0 bytes or has 0 pages.
        PDFSizeLimitError: If file exceeds max_size_mb.
        CorruptPDFError: If the PDF structure is corrupt or cannot be parsed.
    """
    validated_path = validate_pdf_file(file_path, max_size_mb=max_size_mb)
    filename = validated_path.name

    try:
        doc = pymupdf.open(str(validated_path))
    except pymupdf.EmptyFileError as e:
        raise EmptyPDFError(f"Cannot open empty PDF file: {filename}") from e
    except pymupdf.FileDataError as e:
        raise CorruptPDFError(f"Corrupt or invalid PDF file data in '{filename}': {e}") from e
    except Exception as e:
        raise CorruptPDFError(f"Failed to open PDF document '{filename}': {e}") from e

    try:
        if doc.is_encrypted:
            # Attempt to authenticate with empty password (default for some encrypted docs)
            if doc.authenticate("") == 0:
                raise InvalidPDFError(f"PDF document '{filename}' is encrypted and password-protected.")

        total_pages = doc.page_count
        if total_pages == 0:
            raise EmptyPDFError(f"PDF document '{filename}' contains 0 pages.")

        page_records: List[Dict[str, Any]] = []

        for page_idx in range(total_pages):
            page_number = page_idx + 1
            try:
                page = doc[page_idx]
                raw_text = page.get_text("text")
            except Exception as e:
                raise CorruptPDFError(
                    f"Error reading page {page_number} in '{filename}': {e}"
                ) from e

            cleaned = clean_text(raw_text)
            is_low_text = len(cleaned) < min_chars_threshold

            record: Dict[str, Any] = {
                "source": filename,
                "page_number": page_number,
                "text": cleaned,
                "warning": SCANNED_PAGE_WARNING if is_low_text else None,
                "is_scanned": is_low_text,
            }
            page_records.append(record)

        return page_records

    finally:
        doc.close()
