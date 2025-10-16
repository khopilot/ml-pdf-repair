#!/usr/bin/env python3
"""
Extract PERFECT training pairs from gold-standard PDF-text pairs.

This script creates the HIGHEST QUALITY training data by extracting text from
corrupted PDFs and matching it with the perfect reference text files.

Gold-standard pairs:
1. wiki_khmer_os.pdf → wiki_khmer_os.txt
2. ams_khmer_os_battambang.pdf → ams_khmer_os_battambang.txt
3. cambonomist_kantumruy.pdf → cambonomist_kantumruy.txt
4. lomhe_krasar.pdf → lomhe_krasar.txt
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import unicodedata
import re

try:
    import pdfplumber
    import PyPDF2
    from tqdm import tqdm
    import Levenshtein
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pdfplumber PyPDF2 tqdm python-Levenshtein")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoldPairExtractor:
    """Extract training pairs from gold-standard PDF-text pairs."""

    # Gold-standard file pairs
    GOLD_PAIRS = [
        ("wiki_khmer_os.pdf", "wiki_khmer_os.txt"),
        ("ams_khmer_os_battambang.pdf", "ams_khmer_os_battambang.txt"),
        ("cambonomist_kantumruy.pdf", "cambonomist_kantumruy.txt"),
        ("lomhe_krasar.pdf", "lomhe_krasar.txt"),
    ]

    def __init__(
        self,
        pdf_dir: Path,
        txt_dir: Path,
        output_dir: Path,
        max_pages: int = None,
        chunk_size: int = 500,
        min_khmer_ratio: float = 0.80
    ):
        self.pdf_dir = Path(pdf_dir)
        self.txt_dir = Path(txt_dir)
        self.output_dir = Path(output_dir)
        self.max_pages = max_pages
        self.chunk_size = chunk_size
        self.min_khmer_ratio = min_khmer_ratio

        # Statistics
        self.stats = {
            "total_pages_processed": 0,
            "pairs_created": 0,
            "pairs_rejected": 0,
            "avg_cer": 0.0,
            "by_source": {}
        }

    @staticmethod
    def is_khmer_char(char: str) -> bool:
        """Check if character is in Khmer Unicode block."""
        code = ord(char)
        return 0x1780 <= code <= 0x17FF

    def get_khmer_ratio(self, text: str) -> float:
        """Calculate ratio of Khmer characters in text."""
        if not text:
            return 0.0

        khmer_count = sum(1 for c in text if self.is_khmer_char(c))
        total_chars = len([c for c in text if not c.isspace()])

        return khmer_count / total_chars if total_chars > 0 else 0.0

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def extract_pdf_page(self, pdf_path: Path, page_num: int) -> str:
        """Extract text from a single PDF page using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return ""

                page = pdf.pages[page_num]
                text = page.extract_text() or ""
                return self.normalize_text(text)
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path} page {page_num}: {e}")

            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    if page_num >= len(reader.pages):
                        return ""

                    page = reader.pages[page_num]
                    text = page.extract_text() or ""
                    return self.normalize_text(text)
            except Exception as e2:
                logger.error(f"PyPDF2 also failed for {pdf_path} page {page_num}: {e2}")
                return ""

    def read_reference_text(self, txt_path: Path) -> List[str]:
        """Read reference text and split into chunks."""
        logger.info(f"Reading reference text: {txt_path}")

        with open(txt_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        # Split by paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', full_text)

        # Clean paragraphs
        cleaned = []
        for p in paragraphs:
            p = self.normalize_text(p)
            if p and len(p) > 50:  # Minimum length
                cleaned.append(p)

        logger.info(f"Found {len(cleaned)} paragraphs in reference text")
        return cleaned

    def find_best_match(
        self,
        corrupted: str,
        reference_chunks: List[str],
        search_window: int = 100
    ) -> Tuple[str, float]:
        """Find best matching reference chunk for corrupted text."""
        if not corrupted or len(corrupted) < 20:
            return "", 1.0

        best_match = ""
        best_cer = 1.0

        # Search in a window of chunks
        for ref_chunk in reference_chunks[:search_window]:
            # Calculate Character Error Rate
            distance = Levenshtein.distance(corrupted, ref_chunk)
            cer = distance / max(len(corrupted), len(ref_chunk))

            if cer < best_cer:
                best_cer = cer
                best_match = ref_chunk

        return best_match, best_cer

    def is_valid_pair(self, corrupted: str, clean: str) -> Tuple[bool, str]:
        """Validate a training pair."""
        # Check minimum length
        if len(corrupted) < 20 or len(clean) < 20:
            return False, "Text too short"

        # Check Khmer ratio for corrupted text
        corrupted_khmer = self.get_khmer_ratio(corrupted)
        if corrupted_khmer < self.min_khmer_ratio:
            return False, f"Corrupted Khmer ratio too low: {corrupted_khmer:.2%}"

        # Check Khmer ratio for clean text (should be very high)
        clean_khmer = self.get_khmer_ratio(clean)
        if clean_khmer < 0.90:
            return False, f"Clean Khmer ratio too low: {clean_khmer:.2%}"

        # Check CER (corrupted should be different from clean)
        distance = Levenshtein.distance(corrupted, clean)
        cer = distance / max(len(corrupted), len(clean))

        if cer < 0.05:
            return False, f"Texts too similar (CER: {cer:.2%}) - not corrupted enough"

        if cer > 0.95:
            return False, f"Texts too different (CER: {cer:.2%}) - bad match"

        return True, f"Valid pair (CER: {cer:.2%})"

    def extract_pairs_from_file_pair(
        self,
        pdf_name: str,
        txt_name: str
    ) -> List[Dict]:
        """Extract training pairs from one PDF-text file pair."""
        pdf_path = self.pdf_dir / pdf_name
        txt_path = self.txt_dir / txt_name

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_name} → {txt_name}")
        logger.info(f"{'='*60}")

        # Check files exist
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return []

        if not txt_path.exists():
            logger.error(f"Text not found: {txt_path}")
            return []

        # Read reference text
        reference_chunks = self.read_reference_text(txt_path)

        # Get total pages
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
        except:
            logger.error(f"Cannot open PDF: {pdf_path}")
            return []

        # Limit pages if requested
        pages_to_process = min(total_pages, self.max_pages or total_pages)
        logger.info(f"Processing {pages_to_process} of {total_pages} pages")

        pairs = []
        source_stats = {
            "total_pages": pages_to_process,
            "pairs_created": 0,
            "pairs_rejected": 0,
            "avg_cer": 0.0
        }

        # Process each page
        chunk_idx = 0
        total_cer = 0.0

        for page_num in tqdm(range(pages_to_process), desc=f"Extracting {pdf_name}"):
            # Extract corrupted text from PDF
            corrupted_text = self.extract_pdf_page(pdf_path, page_num)

            if not corrupted_text:
                logger.debug(f"No text on page {page_num}")
                continue

            # Find matching clean text from reference
            if chunk_idx >= len(reference_chunks):
                logger.warning(f"Ran out of reference chunks at page {page_num}")
                break

            # Use sequential chunks (assume PDF pages match text order)
            clean_text = reference_chunks[chunk_idx]
            chunk_idx += 1

            # Calculate CER
            distance = Levenshtein.distance(corrupted_text, clean_text)
            cer = distance / max(len(corrupted_text), len(clean_text))

            # Validate pair
            is_valid, reason = self.is_valid_pair(corrupted_text, clean_text)

            if is_valid:
                pairs.append({
                    "input": corrupted_text,
                    "target": clean_text,
                    "metadata": {
                        "source": pdf_name,
                        "page": page_num,
                        "cer": cer,
                        "input_length": len(corrupted_text),
                        "target_length": len(clean_text)
                    }
                })

                source_stats["pairs_created"] += 1
                total_cer += cer

                logger.debug(f"Page {page_num}: Valid pair (CER: {cer:.2%})")
            else:
                source_stats["pairs_rejected"] += 1
                logger.debug(f"Page {page_num}: Rejected - {reason}")

        # Calculate average CER
        if source_stats["pairs_created"] > 0:
            source_stats["avg_cer"] = total_cer / source_stats["pairs_created"]

        logger.info(f"\nResults for {pdf_name}:")
        logger.info(f"  Pairs created: {source_stats['pairs_created']}")
        logger.info(f"  Pairs rejected: {source_stats['pairs_rejected']}")
        logger.info(f"  Average CER: {source_stats['avg_cer']:.2%}")

        self.stats["by_source"][pdf_name] = source_stats

        return pairs

    def split_dataset(
        self,
        pairs: List[Dict],
        train_ratio: float = 0.80,
        val_ratio: float = 0.10
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split pairs into train/val/test sets."""
        import random

        # Shuffle pairs
        random.seed(42)
        random.shuffle(pairs)

        n = len(pairs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = pairs[:train_end]
        val = pairs[train_end:val_end]
        test = pairs[val_end:]

        return train, val, test

    def save_dataset(
        self,
        pairs: List[Dict],
        split_name: str
    ):
        """Save dataset to JSON file."""
        output_file = self.output_dir / f"{split_name}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(pairs)} pairs to {output_file}")

    def run(self):
        """Main extraction pipeline."""
        logger.info("Starting gold-standard pair extraction")
        logger.info(f"PDF directory: {self.pdf_dir}")
        logger.info(f"Text directory: {self.txt_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Max pages per file: {self.max_pages or 'unlimited'}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract pairs from each gold file pair
        all_pairs = []

        for pdf_name, txt_name in self.GOLD_PAIRS:
            pairs = self.extract_pairs_from_file_pair(pdf_name, txt_name)
            all_pairs.extend(pairs)

            self.stats["total_pages_processed"] += self.stats["by_source"].get(
                pdf_name, {}
            ).get("total_pages", 0)

        logger.info(f"\n{'='*60}")
        logger.info("EXTRACTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total pairs extracted: {len(all_pairs)}")

        if not all_pairs:
            logger.error("No pairs extracted! Check your PDF and text files.")
            return

        # Split dataset
        train, val, test = self.split_dataset(all_pairs)

        logger.info(f"Dataset split:")
        logger.info(f"  Train: {len(train)} pairs")
        logger.info(f"  Val:   {len(val)} pairs")
        logger.info(f"  Test:  {len(test)} pairs")

        # Save datasets
        self.save_dataset(train, "train")
        self.save_dataset(val, "val")
        self.save_dataset(test, "test")

        # Calculate and save statistics
        total_cer = sum(
            stats["avg_cer"] * stats["pairs_created"]
            for stats in self.stats["by_source"].values()
            if stats["pairs_created"] > 0
        )
        total_pairs = sum(
            stats["pairs_created"]
            for stats in self.stats["by_source"].values()
        )

        self.stats["pairs_created"] = total_pairs
        self.stats["avg_cer"] = total_cer / total_pairs if total_pairs > 0 else 0.0

        stats_file = self.output_dir / "extraction_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        logger.info(f"\nStatistics saved to {stats_file}")
        logger.info(f"Overall average CER: {self.stats['avg_cer']:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract gold-standard training pairs from perfect PDF-text pairs"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("../../corrupted_pdf_to_exctratc_text_for_comparaison"),
        help="Directory containing corrupted PDF files"
    )
    parser.add_argument(
        "--txt-dir",
        type=Path,
        default=Path("../../correct_txt"),
        help="Directory containing reference text files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for training pairs"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to extract per PDF (default: all)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for text matching"
    )
    parser.add_argument(
        "--min-khmer-ratio",
        type=float,
        default=0.80,
        help="Minimum Khmer character ratio (0.0-1.0)"
    )

    args = parser.parse_args()

    extractor = GoldPairExtractor(
        pdf_dir=args.pdf_dir,
        txt_dir=args.txt_dir,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        chunk_size=args.chunk_size,
        min_khmer_ratio=args.min_khmer_ratio
    )

    extractor.run()


if __name__ == "__main__":
    main()
