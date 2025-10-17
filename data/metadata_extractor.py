#!/usr/bin/env python3
"""
PDF Metadata Extractor for Enhanced Training Pairs

Extracts font information, ToUnicode status, and other PDF metadata
for enriching training datasets.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import subprocess

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False


logger = logging.getLogger(__name__)


class PDFMetadataExtractor:
    """Extract metadata from PDF files for training pair enrichment."""

    @staticmethod
    def extract_font_names(pdf_path: Path, page_num: Optional[int] = None) -> List[str]:
        """
        Extract font names from PDF.

        Args:
            pdf_path: Path to PDF file
            page_num: Specific page (None = all pages)

        Returns:
            List of unique font names
        """
        font_names = set()

        # Try PyMuPDF first (faster)
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(str(pdf_path))
                pages = [page_num] if page_num is not None else range(len(doc))

                for pg_num in pages:
                    if pg_num < len(doc):
                        page = doc[pg_num]
                        # Get font list from page
                        for font in page.get_fonts():
                            # font = (xref, name, type, encoding, ...)
                            if len(font) > 3:
                                font_name = font[3]  # Font name
                                if font_name:
                                    # Clean font name (remove subset prefix)
                                    clean_name = font_name.split('+')[-1]
                                    font_names.add(clean_name)

                doc.close()
                return sorted(list(font_names))
            except Exception as e:
                logger.warning(f"PyMuPDF font extraction failed: {e}")

        # Fallback to pdffonts if available
        try:
            result = subprocess.run(
                ['pdffonts', str(pdf_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[2:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if parts:
                        font_name = parts[0]
                        # Clean font name
                        clean_name = font_name.split('+')[-1]
                        font_names.add(clean_name)
                return sorted(list(font_names))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        logger.warning(f"Could not extract fonts from {pdf_path}")
        return []

    @staticmethod
    def check_tounicode_cmap(pdf_path: Path, page_num: int = 0) -> Optional[bool]:
        """
        Check if PDF fonts have ToUnicode CMaps.

        Args:
            pdf_path: Path to PDF file
            page_num: Page to check

        Returns:
            True if has ToUnicode, False if missing, None if unknown
        """
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(str(pdf_path))
                if page_num < len(doc):
                    page = doc[page_num]

                    # Get font objects
                    for font in page.get_fonts():
                        xref = font[0]
                        # Get font object
                        font_obj = doc.xref_object(xref)

                        # Check for ToUnicode stream
                        if '/ToUnicode' in font_obj:
                            doc.close()
                            return True

                doc.close()
                # If no ToUnicode found in any font
                return False
            except Exception as e:
                logger.warning(f"ToUnicode check failed: {e}")

        # Fallback: check with pdffonts
        try:
            result = subprocess.run(
                ['pdffonts', str(pdf_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # pdffonts output shows 'yes'/'no' in ToUnicode column
                lines = result.stdout.strip().split('\n')
                if len(lines) > 2:
                    # Header line contains 'toUni' or similar
                    header = lines[1]
                    data_line = lines[2]

                    # Find ToUnicode column (usually last or second-to-last)
                    if 'yes' in data_line.lower():
                        return True
                    elif 'no' in data_line.lower():
                        return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None  # Unknown

    @staticmethod
    def detect_extraction_method(pdf_text: str, method_name: str) -> str:
        """
        Validate extraction method by checking output quality.

        Args:
            pdf_text: Extracted text
            method_name: Method used (pdfminer, pymupdf, etc.)

        Returns:
            Validated method name or 'unknown'
        """
        if not pdf_text or len(pdf_text) < 10:
            return "failed"

        # Check if text contains mostly Khmer
        khmer_chars = sum(
            1 for c in pdf_text
            if 0x1780 <= ord(c) <= 0x17FF
        )

        if khmer_chars > len(pdf_text) * 0.3:
            return method_name

        # Check for PUA characters (indicates need for mapping)
        pua_chars = sum(
            1 for c in pdf_text
            if 0xE000 <= ord(c) <= 0xF8FF
        )

        if pua_chars > len(pdf_text) * 0.2:
            return f"{method_name}_pua"

        return method_name

    @staticmethod
    def get_pdf_info(pdf_path: Path) -> Dict[str, any]:
        """
        Get comprehensive PDF metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with PDF metadata
        """
        info = {
            'page_count': 0,
            'fonts': [],
            'has_tounicode': None,
            'producer': None,
            'creator': None,
        }

        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(str(pdf_path))
                info['page_count'] = len(doc)

                # Get metadata
                metadata = doc.metadata
                if metadata:
                    info['producer'] = metadata.get('producer')
                    info['creator'] = metadata.get('creator')

                # Get fonts from first page
                if len(doc) > 0:
                    info['fonts'] = PDFMetadataExtractor.extract_font_names(pdf_path, 0)
                    info['has_tounicode'] = PDFMetadataExtractor.check_tounicode_cmap(pdf_path, 0)

                doc.close()
            except Exception as e:
                logger.error(f"Failed to get PDF info from {pdf_path}: {e}")

        return info


def enrich_training_pair_metadata(
    input_text: str,
    target_text: str,
    pdf_path: Optional[Path] = None,
    page_num: Optional[int] = None,
    extraction_method: Optional[str] = None
) -> Dict:
    """
    Extract all metadata for a training pair.

    Args:
        input_text: Corrupted input text
        target_text: Ground truth text
        pdf_path: Source PDF file
        page_num: Page number
        extraction_method: Method used for extraction

    Returns:
        Dictionary of metadata ready for EnhancedMetadata
    """
    metadata = {}

    if pdf_path and pdf_path.exists():
        # Extract fonts
        fonts = PDFMetadataExtractor.extract_font_names(pdf_path, page_num)
        metadata['font_names'] = fonts

        # Check ToUnicode
        has_tounicode = PDFMetadataExtractor.check_tounicode_cmap(pdf_path, page_num or 0)
        metadata['to_unicode'] = has_tounicode

        # Validate extraction method
        if extraction_method:
            validated_method = PDFMetadataExtractor.detect_extraction_method(
                input_text, extraction_method
            )
            metadata['extraction'] = validated_method
        else:
            metadata['extraction'] = None

    return metadata


if __name__ == "__main__":
    # Test metadata extraction
    import sys

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])

        print(f"Extracting metadata from: {pdf_path}")
        print("="*60)

        # Get comprehensive info
        info = PDFMetadataExtractor.get_pdf_info(pdf_path)

        print(f"\nPage count: {info['page_count']}")
        print(f"Fonts: {', '.join(info['fonts']) if info['fonts'] else 'None found'}")
        print(f"Has ToUnicode: {info['has_tounicode']}")
        print(f"Producer: {info['producer']}")
        print(f"Creator: {info['creator']}")

        # Check specific page
        if info['page_count'] > 0:
            print(f"\nPage 0 fonts: {PDFMetadataExtractor.extract_font_names(pdf_path, 0)}")
            print(f"Page 0 ToUnicode: {PDFMetadataExtractor.check_tounicode_cmap(pdf_path, 0)}")

    else:
        print("Usage: python metadata_extractor.py <pdf_file>")
        print("\nExample:")
        print("python metadata_extractor.py ../corrupted_pdf_to_exctratc_text_for_comparaison/wiki_khmer_os.pdf")
