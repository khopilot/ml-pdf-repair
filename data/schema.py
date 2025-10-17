#!/usr/bin/env python3
"""
Enhanced Dataset Schema for Khmer PDF Correction

Defines the standard metadata-rich format for all training datasets.

Standard Format:
{
  "input": "corrupted text",
  "target": "corrected text",
  "lang": "km",
  "metadata": {
    "source_pdf": "sha1:...",
    "page": 1,
    "bbox_spans": false,
    "extraction": "pdfminer|pymupdf|poppler|tesseract",
    "to_unicode": true,
    "font_names": ["KhmerOS"],
    "is_synthetic": false,
    "noise_tags": ["space_insertion", "char_split", "mark_reorder"],
    "input_length": 1000,
    "target_length": 1000,
    "length_ratio": 1.0,
    "input_khmer_ratio": 0.95,
    "target_khmer_ratio": 0.95,
    "has_pua": false,
    "is_truncated": false,
    "truncation_limit": null,
    "expected_length": 1000,
    "ground_truth_source": "manual_correction_v1",
    "cer_estimate": null
  },
  "split": "train|val|test"
}
"""

import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Noise pattern definitions
NOISE_PATTERNS = {
    "space_insertion": "Extra spaces inserted (e.g., 'ក ្មែ' → 'ខ្មែរ')",
    "space_deletion": "Missing spaces between words",
    "char_split": "Characters split apart (e.g., 'សា ច់' → 'សាច់')",
    "mark_reorder": "Diacritic/vowel marks in wrong order",
    "mark_deletion": "Missing diacritic marks",
    "mark_duplication": "Duplicate diacritic marks",
    "char_substitution": "Wrong character substitution",
    "newline_noise": "Incorrect newline placement",
    "pua_chars": "Private Use Area characters (legacy encoding)",
    "latin_noise": "Unexpected Latin characters",
    "digit_noise": "Digit recognition errors",
    "coeng_error": "Subscript consonant (coeng) errors",
}


@dataclass
class EnhancedMetadata:
    """Enhanced metadata for training pairs."""

    # Source tracking
    source_pdf: Optional[str] = None  # SHA1 hash or filename
    page: Optional[int] = None
    bbox_spans: bool = False

    # Extraction metadata
    extraction: Optional[str] = None  # pdfminer, pymupdf, poppler, tesseract
    to_unicode: Optional[bool] = None  # Has ToUnicode CMap
    font_names: List[str] = field(default_factory=list)

    # Data provenance
    is_synthetic: bool = False  # Artificially generated corruption
    noise_tags: List[str] = field(default_factory=list)

    # Quality metrics
    input_length: Optional[int] = None
    target_length: Optional[int] = None
    length_ratio: Optional[float] = None
    input_khmer_ratio: Optional[float] = None
    target_khmer_ratio: Optional[float] = None
    has_pua: bool = False

    # Truncation tracking
    is_truncated: bool = False
    truncation_limit: Optional[int] = None
    expected_length: Optional[int] = None

    # Ground truth info
    ground_truth_source: Optional[str] = None
    cer_estimate: Optional[float] = None

    # Additional custom metadata
    original_source: Optional[str] = None  # Dataset name when combining
    custom: Dict = field(default_factory=dict)


@dataclass
class EnhancedTrainingPair:
    """Enhanced training pair with metadata."""
    input: str
    target: str
    lang: str = "km"
    metadata: EnhancedMetadata = field(default_factory=EnhancedMetadata)
    split: str = "train"  # train, val, test

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "input": self.input,
            "target": self.target,
            "lang": self.lang,
            "metadata": asdict(self.metadata),
            "split": self.split
        }


class NoiseDetector:
    """Detect noise patterns by comparing input and target."""

    @staticmethod
    def detect_space_insertion(input_text: str, target_text: str) -> bool:
        """Detect extra spaces in input compared to target."""
        input_spaces = input_text.count(' ')
        target_spaces = target_text.count(' ')
        return input_spaces > target_spaces * 1.3

    @staticmethod
    def detect_space_deletion(input_text: str, target_text: str) -> bool:
        """Detect missing spaces in input."""
        input_spaces = input_text.count(' ')
        target_spaces = target_text.count(' ')
        return target_spaces > input_spaces * 1.3

    @staticmethod
    def detect_char_split(text: str) -> bool:
        """Detect split Khmer characters (space between base + diacritic)."""
        # Pattern: Khmer base char + space + combining mark
        pattern = r'[\u1780-\u17B3]\s+[\u17B4-\u17DD]'
        return bool(re.search(pattern, text))

    @staticmethod
    def detect_mark_errors(input_text: str, target_text: str) -> bool:
        """Detect diacritic mark errors."""
        input_marks = sum(1 for c in input_text if '\u17B4' <= c <= '\u17DD')
        target_marks = sum(1 for c in target_text if '\u17B4' <= c <= '\u17DD')
        return abs(input_marks - target_marks) > min(input_marks, target_marks) * 0.1

    @staticmethod
    def detect_newline_noise(text: str) -> bool:
        """Detect excessive newlines."""
        return '\n\n' in text or text.count('\n') > len(text) / 50

    @staticmethod
    def detect_pua_chars(text: str) -> bool:
        """Detect Private Use Area characters (legacy fonts)."""
        return any('\uE000' <= c <= '\uF8FF' for c in text)

    @staticmethod
    def detect_latin_noise(text: str, khmer_ratio: float) -> bool:
        """Detect unexpected Latin characters in primarily Khmer text."""
        if khmer_ratio < 0.5:
            return False
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        return latin_chars > len(text) * 0.1

    @staticmethod
    def detect_coeng_error(input_text: str, target_text: str) -> bool:
        """Detect subscript consonant (coeng) errors."""
        input_coeng = input_text.count('\u17D2')
        target_coeng = target_text.count('\u17D2')
        return abs(input_coeng - target_coeng) > 0

    @classmethod
    def detect_all_noise(cls, input_text: str, target_text: str,
                        input_khmer_ratio: float, target_khmer_ratio: float) -> List[str]:
        """Detect all noise patterns."""
        noise_tags = []

        if cls.detect_space_insertion(input_text, target_text):
            noise_tags.append("space_insertion")

        if cls.detect_space_deletion(input_text, target_text):
            noise_tags.append("space_deletion")

        if cls.detect_char_split(input_text):
            noise_tags.append("char_split")

        if cls.detect_mark_errors(input_text, target_text):
            noise_tags.append("mark_reorder")

        if cls.detect_newline_noise(input_text):
            noise_tags.append("newline_noise")

        if cls.detect_pua_chars(input_text):
            noise_tags.append("pua_chars")

        if cls.detect_latin_noise(input_text, input_khmer_ratio):
            noise_tags.append("latin_noise")

        if cls.detect_coeng_error(input_text, target_text):
            noise_tags.append("coeng_error")

        # Check for character substitutions (length similar but different content)
        if abs(len(input_text) - len(target_text)) < min(len(input_text), len(target_text)) * 0.1:
            if input_text != target_text:
                noise_tags.append("char_substitution")

        return noise_tags


class MetricsCalculator:
    """Calculate quality metrics for training pairs."""

    @staticmethod
    def get_khmer_ratio(text: str) -> float:
        """Calculate ratio of Khmer characters."""
        if not text:
            return 0.0
        khmer_chars = sum(
            1 for c in text
            if 0x1780 <= ord(c) <= 0x17FF or 0x19E0 <= ord(c) <= 0x19FF
        )
        return khmer_chars / len(text)

    @staticmethod
    def calculate_length_ratio(input_text: str, target_text: str) -> float:
        """Calculate length ratio (target/input)."""
        if len(input_text) == 0:
            return 0.0
        return len(target_text) / len(input_text)

    @staticmethod
    def has_pua_characters(text: str) -> bool:
        """Check for Private Use Area characters."""
        return any('\uE000' <= c <= '\uF8FF' for c in text)

    @staticmethod
    def detect_truncation(input_len: int, target_len: int,
                         truncation_threshold: int = 800) -> Tuple[bool, Optional[int]]:
        """Detect if target is artificially truncated."""
        is_truncated = (
            target_len == truncation_threshold and
            input_len > target_len * 1.2
        )
        return is_truncated, truncation_threshold if is_truncated else None


class PDFHashGenerator:
    """Generate SHA1 hashes for PDF source tracking."""

    @staticmethod
    def get_file_hash(pdf_path: Path) -> str:
        """Generate SHA1 hash of PDF file."""
        sha1 = hashlib.sha1()
        try:
            with open(pdf_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha1.update(chunk)
            return f"sha1:{sha1.hexdigest()}"
        except Exception:
            # Fallback to filename if file not accessible
            return f"file:{pdf_path.name}"

    @staticmethod
    def get_text_hash(text: str) -> str:
        """Generate hash of text content."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()


def validate_training_pair(pair: Dict) -> Tuple[bool, List[str]]:
    """
    Validate training pair against enhanced schema.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Required fields
    if "input" not in pair:
        errors.append("Missing required field: 'input'")
    if "target" not in pair:
        errors.append("Missing required field: 'target'")

    # Language field
    if "lang" not in pair:
        errors.append("Missing 'lang' field")
    elif pair["lang"] != "km":
        errors.append(f"Unexpected language: {pair['lang']} (expected 'km')")

    # Metadata validation
    if "metadata" not in pair:
        errors.append("Missing 'metadata' field")
    else:
        metadata = pair["metadata"]

        # Check critical fields exist (can be null)
        critical_fields = [
            "extraction", "noise_tags", "input_length", "target_length",
            "length_ratio", "is_truncated"
        ]
        for field in critical_fields:
            if field not in metadata:
                errors.append(f"Missing metadata.{field}")

        # Validate noise_tags format
        if "noise_tags" in metadata and not isinstance(metadata["noise_tags"], list):
            errors.append("metadata.noise_tags must be a list")

    # Split validation
    if "split" not in pair:
        errors.append("Missing 'split' field")
    elif pair["split"] not in ["train", "val", "test"]:
        errors.append(f"Invalid split: {pair['split']}")

    return len(errors) == 0, errors


def create_enhanced_pair(
    input_text: str,
    target_text: str,
    pdf_path: Optional[Path] = None,
    page_num: Optional[int] = None,
    extraction_method: Optional[str] = None,
    font_names: Optional[List[str]] = None,
    is_synthetic: bool = False,
    ground_truth_source: Optional[str] = None,
    split: str = "train",
    **kwargs
) -> EnhancedTrainingPair:
    """
    Create enhanced training pair with auto-detected metadata.

    Args:
        input_text: Corrupted text
        target_text: Corrected text
        pdf_path: Source PDF file (optional)
        page_num: Page number (optional)
        extraction_method: Extraction method used
        font_names: List of font names from PDF
        is_synthetic: Whether corruption is artificial
        ground_truth_source: Source of ground truth
        split: Dataset split (train/val/test)
        **kwargs: Additional metadata fields

    Returns:
        EnhancedTrainingPair with auto-detected metadata
    """
    # Calculate metrics
    input_length = len(input_text)
    target_length = len(target_text)
    length_ratio = MetricsCalculator.calculate_length_ratio(input_text, target_text)
    input_khmer_ratio = MetricsCalculator.get_khmer_ratio(input_text)
    target_khmer_ratio = MetricsCalculator.get_khmer_ratio(target_text)
    has_pua = MetricsCalculator.has_pua_characters(input_text)

    # Detect truncation
    is_truncated, truncation_limit = MetricsCalculator.detect_truncation(
        input_length, target_length
    )

    # Detect noise patterns
    noise_tags = NoiseDetector.detect_all_noise(
        input_text, target_text, input_khmer_ratio, target_khmer_ratio
    )

    # Generate PDF hash if path provided
    source_pdf = None
    if pdf_path:
        source_pdf = PDFHashGenerator.get_file_hash(pdf_path)

    # Create metadata
    metadata = EnhancedMetadata(
        source_pdf=source_pdf,
        page=page_num,
        extraction=extraction_method,
        font_names=font_names or [],
        is_synthetic=is_synthetic,
        noise_tags=noise_tags,
        input_length=input_length,
        target_length=target_length,
        length_ratio=round(length_ratio, 3),
        input_khmer_ratio=round(input_khmer_ratio, 3),
        target_khmer_ratio=round(target_khmer_ratio, 3),
        has_pua=has_pua,
        is_truncated=is_truncated,
        truncation_limit=truncation_limit,
        expected_length=input_length if is_truncated else target_length,
        ground_truth_source=ground_truth_source,
        custom=kwargs
    )

    return EnhancedTrainingPair(
        input=input_text,
        target=target_text,
        lang="km",
        metadata=metadata,
        split=split
    )


if __name__ == "__main__":
    # Test schema
    print("Testing Enhanced Dataset Schema...")

    # Create test pair
    test_pair = create_enhanced_pair(
        input_text="ក ្នុងការដោះស ្រាយ",
        target_text="ក្នុងការដោះស្រាយ",
        extraction_method="pdfminer",
        ground_truth_source="manual_correction",
        split="train"
    )

    # Convert to dict
    pair_dict = test_pair.to_dict()

    # Validate
    is_valid, errors = validate_training_pair(pair_dict)

    print(f"\nValidation: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if errors:
        print("Errors:", errors)

    print(f"\nDetected noise tags: {pair_dict['metadata']['noise_tags']}")
    print(f"Length ratio: {pair_dict['metadata']['length_ratio']}")
    print(f"Khmer ratio: {pair_dict['metadata']['target_khmer_ratio']}")

    import json
    print("\nSample output:")
    print(json.dumps(pair_dict, ensure_ascii=False, indent=2))
