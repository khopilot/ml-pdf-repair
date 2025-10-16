#!/usr/bin/env python3
"""
Training Data Collection for Khmer PDF Corruption Correction

Extracts PDF→OCR paired sequences for training character-level models.

Usage:
    python collect_training_pairs.py \\
        --pdf-dir ../corrupted_pdf_to_exctratc_text_for_comparaison \\
        --output-dir data/training_pairs \\
        --num-pages 500 \\
        --ocr-lang khm
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import subprocess
import unicodedata

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    print("Warning: pdfminer.six not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available")

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not available for OCR")


@dataclass
class TrainingPair:
    """Single training example: corrupted PDF text → correct OCR text."""
    input_text: str          # Corrupted text from PDF extraction
    target_text: str         # Ground truth from OCR
    page_num: int            # Page number in PDF
    pdf_name: str            # Source PDF file name
    method: str              # Extraction method used (pdfminer, pymupdf)
    input_length: int        # Number of characters in input
    target_length: int       # Number of characters in target
    khmer_ratio: float       # Ratio of Khmer characters in text


class TrainingDataCollector:
    """
    Collects corrupted→correct text pairs from PDFs.

    Uses PDF extraction (corrupted) and OCR (ground truth) to create
    character-level training data for seq2seq models.
    """

    def __init__(self,
                 pdf_dir: Path,
                 output_dir: Path,
                 ocr_lang: str = "khm",
                 ocr_dpi: int = 400,
                 min_khmer_ratio: float = 0.3):
        """
        Initialize data collector.

        Args:
            pdf_dir: Directory containing PDFs
            output_dir: Where to save training pairs
            ocr_lang: Tesseract language code
            ocr_dpi: OCR rendering DPI
            min_khmer_ratio: Minimum Khmer character ratio to include
        """
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.ocr_lang = ocr_lang
        self.ocr_dpi = ocr_dpi
        self.min_khmer_ratio = min_khmer_ratio

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_pdf_text(self, pdf_path: Path, page_num: int) -> Tuple[str, str]:
        """
        Extract text from PDF page using both pdfminer and pymupdf.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            (pdfminer_text, pymupdf_text) tuple
        """
        pdfminer_text = ""
        pymupdf_text = ""

        # Try pdfminer
        if PDFMINER_AVAILABLE:
            try:
                # Extract single page
                full_text = extract_text(
                    str(pdf_path),
                    page_numbers=[page_num],
                    laparams=LAParams()
                )
                pdfminer_text = full_text.strip()
            except Exception as e:
                self.logger.warning(f"pdfminer failed on {pdf_path} page {page_num}: {e}")

        # Try PyMuPDF
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(str(pdf_path))
                if page_num < len(doc):
                    page = doc[page_num]
                    pymupdf_text = page.get_text().strip()
                doc.close()
            except Exception as e:
                self.logger.warning(f"PyMuPDF failed on {pdf_path} page {page_num}: {e}")

        return pdfminer_text, pymupdf_text

    def extract_ocr_text(self, pdf_path: Path, page_num: int) -> str:
        """
        Extract text using OCR (Tesseract) as ground truth.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            OCR extracted text
        """
        if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
            self.logger.error("PyMuPDF and Pillow required for OCR")
            return ""

        try:
            # Render PDF page to image
            doc = fitz.open(str(pdf_path))
            page = doc[page_num]

            # Render at high DPI for OCR
            mat = fitz.Matrix(self.ocr_dpi / 72, self.ocr_dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()

            # Save temporarily
            temp_img = self.output_dir / f"temp_page_{page_num}.png"
            img.save(temp_img)

            # Run Tesseract
            cmd = [
                "tesseract",
                str(temp_img),
                "stdout",
                "-l", self.ocr_lang,
                "--psm", "6"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Clean up
            temp_img.unlink()

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.logger.warning(f"Tesseract failed: {result.stderr}")
                return ""

        except Exception as e:
            self.logger.error(f"OCR extraction failed on {pdf_path} page {page_num}: {e}")
            return ""

    @staticmethod
    def get_khmer_ratio(text: str) -> float:
        """Calculate ratio of Khmer characters in text."""
        if not text:
            return 0.0

        khmer_chars = sum(
            1 for c in text
            if 0x1780 <= ord(c) <= 0x17FF or 0x19E0 <= ord(c) <= 0x19FF
        )

        return khmer_chars / len(text)

    @staticmethod
    def normalize_for_training(text: str) -> str:
        """
        Normalize text for training (minimal normalization).

        - Apply NFC normalization
        - Remove ZWSP (but keep ZWJ/ZWNJ for Khmer shaping)
        - Collapse multiple spaces
        """
        # NFC normalization
        text = unicodedata.normalize('NFC', text)

        # Remove ZWSP only (U+200B)
        text = text.replace('\u200B', '')

        # Collapse multiple spaces/newlines
        text = ' '.join(text.split())

        return text

    def create_training_pair(self,
                            pdf_path: Path,
                            page_num: int) -> Optional[TrainingPair]:
        """
        Create a single training pair for one PDF page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            TrainingPair or None if extraction failed
        """
        self.logger.info(f"Processing {pdf_path.name} page {page_num + 1}")

        # Extract PDF text (corrupted)
        pdfminer_text, pymupdf_text = self.extract_pdf_text(pdf_path, page_num)

        # Choose best PDF extraction (longest with highest Khmer ratio)
        pdf_candidates = [
            ("pdfminer", pdfminer_text),
            ("pymupdf", pymupdf_text)
        ]

        pdf_candidates = [
            (method, text) for method, text in pdf_candidates
            if text and len(text) > 50  # Minimum length
        ]

        if not pdf_candidates:
            self.logger.warning(f"No valid PDF extraction for {pdf_path.name} page {page_num + 1}")
            return None

        # Select best by Khmer ratio
        best_method, best_pdf_text = max(
            pdf_candidates,
            key=lambda x: self.get_khmer_ratio(x[1])
        )

        # Extract OCR text (ground truth)
        ocr_text = self.extract_ocr_text(pdf_path, page_num)

        if not ocr_text or len(ocr_text) < 50:
            self.logger.warning(f"OCR failed for {pdf_path.name} page {page_num + 1}")
            return None

        # Normalize both
        input_text = self.normalize_for_training(best_pdf_text)
        target_text = self.normalize_for_training(ocr_text)

        # Check Khmer ratio
        khmer_ratio = self.get_khmer_ratio(target_text)
        if khmer_ratio < self.min_khmer_ratio:
            self.logger.debug(
                f"Skipping page {page_num + 1}: Khmer ratio {khmer_ratio:.2%} "
                f"< threshold {self.min_khmer_ratio:.2%}"
            )
            return None

        return TrainingPair(
            input_text=input_text,
            target_text=target_text,
            page_num=page_num,
            pdf_name=pdf_path.name,
            method=best_method,
            input_length=len(input_text),
            target_length=len(target_text),
            khmer_ratio=khmer_ratio
        )

    def collect_all_pairs(self,
                         max_pages: Optional[int] = None,
                         pages_per_pdf: Optional[int] = None) -> List[TrainingPair]:
        """
        Collect training pairs from all PDFs in directory.

        Args:
            max_pages: Maximum total pages to process
            pages_per_pdf: Maximum pages per PDF (evenly distributed)

        Returns:
            List of TrainingPair objects
        """
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            self.logger.error(f"No PDF files found in {self.pdf_dir}")
            return []

        self.logger.info(f"Found {len(pdf_files)} PDF files")

        all_pairs = []

        # Determine pages per PDF for even distribution
        if pages_per_pdf is None and max_pages is not None:
            pages_per_pdf = max_pages // len(pdf_files)

        for pdf_path in pdf_files:
            self.logger.info(f"Processing {pdf_path.name}")

            # Get total pages
            try:
                if PYMUPDF_AVAILABLE:
                    doc = fitz.open(str(pdf_path))
                    total_pages = len(doc)
                    doc.close()
                else:
                    total_pages = 100  # Fallback estimate
            except Exception as e:
                self.logger.error(f"Failed to get page count for {pdf_path}: {e}")
                continue

            # Determine pages to process for this PDF
            if pages_per_pdf:
                pages_to_process = min(pages_per_pdf, total_pages)
            elif max_pages:
                remaining = max_pages - len(all_pairs)
                pages_to_process = min(remaining, total_pages)
            else:
                pages_to_process = total_pages

            # Process pages
            for page_num in range(pages_to_process):
                if max_pages and len(all_pairs) >= max_pages:
                    break

                pair = self.create_training_pair(pdf_path, page_num)
                if pair:
                    all_pairs.append(pair)

            self.logger.info(
                f"Collected {len(all_pairs)} pairs so far from {pdf_path.name}"
            )

            if max_pages and len(all_pairs) >= max_pages:
                break

        self.logger.info(f"Total pairs collected: {len(all_pairs)}")
        return all_pairs

    def split_dataset(self,
                     pairs: List[TrainingPair],
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     seed: int = 42) -> Dict[str, List[TrainingPair]]:
        """
        Split pairs into train/val/test sets.

        Stratified by PDF source to ensure all PDFs in each split.

        Args:
            pairs: List of training pairs
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility

        Returns:
            Dict with 'train', 'val', 'test' keys
        """
        import random
        random.seed(seed)

        # Group by PDF
        by_pdf = {}
        for pair in pairs:
            if pair.pdf_name not in by_pdf:
                by_pdf[pair.pdf_name] = []
            by_pdf[pair.pdf_name].append(pair)

        # Split each PDF's pairs
        train_pairs = []
        val_pairs = []
        test_pairs = []

        for pdf_name, pdf_pairs in by_pdf.items():
            random.shuffle(pdf_pairs)

            n = len(pdf_pairs)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_pairs.extend(pdf_pairs[:train_end])
            val_pairs.extend(pdf_pairs[train_end:val_end])
            test_pairs.extend(pdf_pairs[val_end:])

        # Shuffle final splits
        random.shuffle(train_pairs)
        random.shuffle(val_pairs)
        random.shuffle(test_pairs)

        self.logger.info(f"Split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

        return {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }

    def save_dataset(self, split_data: Dict[str, List[TrainingPair]]):
        """
        Save training dataset to JSON files.

        Args:
            split_data: Dict with 'train', 'val', 'test' splits
        """
        for split_name, pairs in split_data.items():
            output_file = self.output_dir / f"{split_name}.json"

            # Convert to dict
            data = [asdict(pair) for pair in pairs]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Saved {len(pairs)} pairs to {output_file}")

        # Save metadata
        metadata = {
            'total_pairs': sum(len(pairs) for pairs in split_data.values()),
            'train_size': len(split_data['train']),
            'val_size': len(split_data['val']),
            'test_size': len(split_data['test']),
            'pdf_sources': list(set(p.pdf_name for pairs in split_data.values() for p in pairs)),
            'avg_input_length': sum(p.input_length for pairs in split_data.values() for p in pairs) / sum(len(pairs) for pairs in split_data.values()),
            'avg_target_length': sum(p.target_length for pairs in split_data.values() for p in pairs) / sum(len(pairs) for pairs in split_data.values()),
            'avg_khmer_ratio': sum(p.khmer_ratio for pairs in split_data.values() for p in pairs) / sum(len(pairs) for pairs in split_data.values()),
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved metadata to {metadata_file}")
        self.logger.info(f"Dataset statistics: {metadata}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect PDF→OCR training pairs for Khmer text correction"
    )

    parser.add_argument(
        '--pdf-dir',
        type=str,
        required=True,
        help='Directory containing source PDFs'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for training data'
    )

    parser.add_argument(
        '--num-pages',
        type=int,
        default=None,
        help='Maximum total pages to process (default: all)'
    )

    parser.add_argument(
        '--pages-per-pdf',
        type=int,
        default=None,
        help='Maximum pages per PDF (default: auto-distribute)'
    )

    parser.add_argument(
        '--ocr-lang',
        type=str,
        default='khm',
        help='Tesseract language code (default: khm)'
    )

    parser.add_argument(
        '--ocr-dpi',
        type=int,
        default=400,
        help='OCR rendering DPI (default: 400)'
    )

    parser.add_argument(
        '--min-khmer-ratio',
        type=float,
        default=0.3,
        help='Minimum Khmer character ratio (default: 0.3)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val/test split (default: 42)'
    )

    args = parser.parse_args()

    # Create collector
    collector = TrainingDataCollector(
        pdf_dir=Path(args.pdf_dir),
        output_dir=Path(args.output_dir),
        ocr_lang=args.ocr_lang,
        ocr_dpi=args.ocr_dpi,
        min_khmer_ratio=args.min_khmer_ratio
    )

    # Collect pairs
    pairs = collector.collect_all_pairs(
        max_pages=args.num_pages,
        pages_per_pdf=args.pages_per_pdf
    )

    if not pairs:
        print("ERROR: No training pairs collected")
        sys.exit(1)

    # Split dataset
    split_data = collector.split_dataset(pairs, seed=args.seed)

    # Save
    collector.save_dataset(split_data)

    print(f"\n✅ Dataset creation complete!")
    print(f"Train: {len(split_data['train'])} pairs")
    print(f"Val: {len(split_data['val'])} pairs")
    print(f"Test: {len(split_data['test'])} pairs")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
