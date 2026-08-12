"""
parser.py — Resume text extraction

Supports PDF, DOCX, and plain TXT formats.
Returns raw text strings that get passed to the embedding and scoring layers.
"""

import os
import logging
from pathlib import Path

import pdfplumber
from docx import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract all text from a text-based PDF file using pdfplumber.

    Note: This does NOT support scanned/image-only PDFs. For those,
    an OCR step (e.g. pytesseract) would be needed.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Concatenated text of all pages, stripped of excess whitespace.
    """
    text_parts = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())
                else:
                    logger.debug(
                        "Page %d of '%s' returned no text (may be image-based).",
                        page_num,
                        file_path,
                    )
    except Exception as exc:
        logger.error("Failed to parse PDF '%s': %s", file_path, exc)
        return ""

    return "\n\n".join(text_parts)


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract all paragraph text from a DOCX file.

    Args:
        file_path: Path to the .docx file.

    Returns:
        Full text joined by newlines.
    """
    try:
        doc = Document(file_path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    except Exception as exc:
        logger.error("Failed to parse DOCX '%s': %s", file_path, exc)
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """
    Read a plain-text resume file.

    Args:
        file_path: Path to the .txt file.

    Returns:
        Raw file content as a string.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception as exc:
        logger.error("Failed to read TXT '%s': %s", file_path, exc)
        return ""


def extract_text(file_path: str) -> str:
    """
    Dispatch to the correct extraction function based on file extension.

    Args:
        file_path: Path to the resume file.

    Returns:
        Extracted text, or an empty string on failure.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        logger.warning("Unsupported file type '%s' — skipping '%s'.", ext, file_path)
        return ""


def load_all_resumes(resume_folder: str) -> dict[str, str]:
    """
    Walk a folder and extract text from every supported resume file.

    Args:
        resume_folder: Path to the directory containing resume files.

    Returns:
        A dict mapping filename → extracted text.
        Files that fail to parse are included with an empty string value.
    """
    folder = Path(resume_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Resume folder not found: {resume_folder}")

    resumes: dict[str, str] = {}

    for entry in sorted(folder.iterdir()):
        if entry.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        logger.info("Parsing resume: %s", entry.name)
        text = extract_text(str(entry))

        if not text:
            logger.warning("No text extracted from '%s' — it will receive a score of 0.", entry.name)

        resumes[entry.name] = text

    if not resumes:
        raise ValueError(f"No supported resume files found in '{resume_folder}'.")

    logger.info("Loaded %d resume(s) from '%s'.", len(resumes), resume_folder)
    return resumes
