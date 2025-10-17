#!/usr/bin/env python3
"""
Convert paired_3k+ dataset to enhanced metadata format.

This dataset contains:
- PDF files: Khmer text with character-level corruptions (diacritic reordering, etc.)
- TXT files: Ground truth correct text

The PDF and TXT files have nearly identical lengths (~0.97 ratio), making this
perfect for character-level correction training.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2

# Import enhanced metadata utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
from schema import create_enhanced_pair, validate_training_pair
from dataclasses import asdict

# Try to import PyMuPDF for font metadata extraction
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Paired3kConverter:
    """Convert paired_3k+ dataset to enhanced format."""

    def __init__(self,
                 pdf_dir: Path,
                 txt_dir: Path,
                 output_dir: Path,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_seed: int = 42):
        self.pdf_dir = Path(pdf_dir)
        self.txt_dir = Path(txt_dir)
        self.output_dir = Path(output_dir)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        random.seed(random_seed)

        self.stats = {
            "total_files": 0,
            "processed": 0,
            "errors": 0,
            "skipped": 0,
            "length_ratios": [],
            "splits": {
                "train": 0,
                "val": 0,
                "test": 0
            }
        }

    def extract_pdf_text(self, pdf_path: Path) -> Optional[str]:
        """Extract text from PDF using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text_parts = []
                for page in reader.pages:
                    text_parts.append(page.extract_text())
                return "".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {e}")
            return None

    def extract_pdf_fonts(self, pdf_path: Path) -> Dict:
        """Extract font metadata from PDF using PyMuPDF."""
        if not FITZ_AVAILABLE:
            return {}

        try:
            doc = fitz.open(str(pdf_path))
            fonts = {}
            has_tounicode = False

            for page_num in range(len(doc)):
                page = doc[page_num]
                font_list = page.get_fonts(full=True)

                for font_info in font_list:
                    font_name = font_info[3] if len(font_info) > 3 else "Unknown"
                    if font_name not in fonts:
                        fonts[font_name] = {"pages": []}
                    if page_num not in fonts[font_name]["pages"]:
                        fonts[font_name]["pages"].append(page_num + 1)

                    # Check if this font has ToUnicode CMap (index 8 in PyMuPDF font info)
                    if len(font_info) > 8 and font_info[8]:
                        has_tounicode = True

            doc.close()
            return {"fonts": fonts, "has_tounicode": has_tounicode}
        except Exception as e:
            logger.warning(f"Could not extract font metadata from {pdf_path.name}: {e}")
            return {}

    def read_txt_file(self, txt_path: Path) -> Optional[str]:
        """Read ground truth text from TXT file."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {txt_path.name}: {e}")
            return None

    def process_pair(self, pdf_path: Path) -> Optional[Dict]:
        """Process a single PDF-TXT pair."""
        # Find corresponding TXT file
        txt_path = self.txt_dir / pdf_path.name.replace('.pdf', '.txt')

        if not txt_path.exists():
            logger.warning(f"No TXT file found for {pdf_path.name}")
            self.stats["skipped"] += 1
            return None

        # Extract corrupted text from PDF
        pdf_text = self.extract_pdf_text(pdf_path)
        if pdf_text is None:
            self.stats["errors"] += 1
            return None

        # Read ground truth from TXT
        txt_content = self.read_txt_file(txt_path)
        if txt_content is None:
            self.stats["errors"] += 1
            return None

        # Skip if either is empty
        if not pdf_text.strip() or not txt_content.strip():
            logger.warning(f"Empty content in {pdf_path.name}")
            self.stats["skipped"] += 1
            return None

        # Track length ratio
        ratio = len(txt_content) / len(pdf_text) if len(pdf_text) > 0 else 0
        self.stats["length_ratios"].append(ratio)

        # Skip if ratio is too far off (likely extraction error)
        if ratio < 0.85 or ratio > 1.15:
            logger.warning(f"Unusual length ratio {ratio:.3f} for {pdf_path.name}")
            self.stats["skipped"] += 1
            return None

        # Extract PDF metadata
        pdf_metadata = self.extract_pdf_fonts(pdf_path)

        # Create enhanced pair
        pair = create_enhanced_pair(
            input_text=pdf_text,
            target_text=txt_content,
            source_pdf=pdf_path.name,
            page=None,  # Multi-page PDFs, all concatenated
            extraction_method="pypdf2",
            font_info=pdf_metadata.get("fonts", {}),
            has_tounicode=pdf_metadata.get("has_tounicode", False),
            is_synthetic=False,
            split="train"  # Will be updated later
        )

        # Convert to dict
        pair_dict = asdict(pair)

        # Validate (skip warnings - (True, []) means passed)
        try:
            is_valid, error_list = validate_training_pair(pair_dict)
            if not is_valid and error_list:
                logger.warning(f"Validation errors for {pdf_path.name}: {error_list}")
        except:
            pass  # Validation is optional

        self.stats["processed"] += 1
        return pair_dict

    def split_data(self, pairs: List[Dict]) -> Dict[str, List[Dict]]:
        """Split data into train/val/test sets."""
        # Shuffle
        random.shuffle(pairs)

        total = len(pairs)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)

        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        test_pairs = pairs[train_size + val_size:]

        # Update split field
        for pair in train_pairs:
            pair["split"] = "train"
        for pair in val_pairs:
            pair["split"] = "val"
        for pair in test_pairs:
            pair["split"] = "test"

        self.stats["splits"]["train"] = len(train_pairs)
        self.stats["splits"]["val"] = len(val_pairs)
        self.stats["splits"]["test"] = len(test_pairs)

        return {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs
        }

    def save_split(self, pairs: List[Dict], split_name: str):
        """Save a dataset split to JSON."""
        output_file = self.output_dir / f"{split_name}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(pairs)} pairs to {output_file}")

    def save_stats(self):
        """Save conversion statistics."""
        stats_file = self.output_dir / "combination_stats.json"

        # Calculate summary stats
        if self.stats["length_ratios"]:
            avg_ratio = sum(self.stats["length_ratios"]) / len(self.stats["length_ratios"])
            min_ratio = min(self.stats["length_ratios"])
            max_ratio = max(self.stats["length_ratios"])
        else:
            avg_ratio = min_ratio = max_ratio = 0

        stats = {
            "total_pairs": self.stats["processed"],
            "train_size": self.stats["splits"]["train"],
            "val_size": self.stats["splits"]["val"],
            "test_size": self.stats["splits"]["test"],
            "errors": self.stats["errors"],
            "skipped": self.stats["skipped"],
            "length_ratio_stats": {
                "mean": round(avg_ratio, 3),
                "min": round(min_ratio, 3),
                "max": round(max_ratio, 3)
            },
            "source": "paired_3k+"
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved statistics to {stats_file}")

    def run(self):
        """Main conversion pipeline."""
        logger.info("="*80)
        logger.info("Converting paired_3k+ dataset to enhanced format")
        logger.info("="*80)
        logger.info(f"PDF directory: {self.pdf_dir}")
        logger.info(f"TXT directory: {self.txt_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Split ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all PDF files
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        self.stats["total_files"] = len(pdf_files)

        logger.info(f"\nFound {len(pdf_files)} PDF files")

        # Process all pairs
        all_pairs = []
        for i, pdf_path in enumerate(pdf_files, 1):
            if i % 100 == 0:
                logger.info(f"Processing {i}/{len(pdf_files)}...")

            pair = self.process_pair(pdf_path)
            if pair:
                all_pairs.append(pair)

        logger.info(f"\nProcessed {self.stats['processed']} pairs successfully")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Skipped: {self.stats['skipped']}")

        if not all_pairs:
            logger.error("No pairs to save!")
            return

        # Split into train/val/test
        logger.info("\nSplitting data...")
        splits = self.split_data(all_pairs)

        # Save splits
        for split_name, pairs in splits.items():
            self.save_split(pairs, split_name)

        # Save statistics
        self.save_stats()

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("CONVERSION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total pairs: {self.stats['processed']}")
        logger.info(f"  Train: {self.stats['splits']['train']} pairs")
        logger.info(f"  Val:   {self.stats['splits']['val']} pairs")
        logger.info(f"  Test:  {self.stats['splits']['test']} pairs")

        if self.stats["length_ratios"]:
            avg_ratio = sum(self.stats["length_ratios"]) / len(self.stats["length_ratios"])
            logger.info(f"\nAverage TXT/PDF length ratio: {avg_ratio:.3f}")

        logger.info(f"\nOutput saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert paired_3k+ dataset to enhanced format"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path(__file__).parent / "paired_3k+" / "pdf",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--txt-dir",
        type=Path,
        default=Path(__file__).parent / "paired_3k+" / "txt",
        help="Directory containing TXT files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "training_pairs_ams_2558p",
        help="Output directory for converted dataset"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    converter = Paired3kConverter(
        pdf_dir=args.pdf_dir,
        txt_dir=args.txt_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    converter.run()


if __name__ == "__main__":
    main()
