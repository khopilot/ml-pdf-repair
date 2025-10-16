#!/usr/bin/env python3
"""
Extract training pairs from forensic recovery directories.

This script uses the pre-extracted page texts and recovered.txt files from
the forensic pipeline to create high-quality training pairs.

Forensic directories:
- cmap_fixed/ (30 pages)
- freq_final/ (300 pages)
- iter1_final/ (300 pages)
- iter1_fuzzy/ (300 pages)
- iter1_wide/ (300 pages)
"""

import argparse
import json
import logging
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import unicodedata
import re

try:
    import Levenshtein
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install python-Levenshtein tqdm")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForensicPairExtractor:
    """Extract training pairs from forensic recovery directories."""

    def __init__(
        self,
        forensic_dir: Path,
        output_dir: Path,
        min_khmer_ratio: float = 0.70,
        max_cer: float = 0.95,
        min_cer: float = 0.05
    ):
        self.forensic_dir = Path(forensic_dir)
        self.output_dir = Path(output_dir)
        self.min_khmer_ratio = min_khmer_ratio
        self.max_cer = max_cer
        self.min_cer = min_cer

        # Statistics
        self.stats = {
            "total_pages_processed": 0,
            "pairs_created": 0,
            "pairs_rejected": 0,
            "avg_cer": 0.0,
            "rejection_reasons": {}
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

    def has_pua_chars(self, text: str) -> bool:
        """Check if text contains Private Use Area characters."""
        for char in text:
            code = ord(char)
            if 0xE000 <= code <= 0xF8FF:  # PUA range
                return True
        return False

    def load_page_scores(self) -> Dict[int, Dict]:
        """Load page scores from diagnostics/page_scores.csv."""
        csv_path = self.forensic_dir / "diagnostics" / "page_scores.csv"

        if not csv_path.exists():
            logger.warning(f"No page_scores.csv found at {csv_path}")
            return {}

        page_scores = {}

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                page_num = int(row['page_num'])
                method = row['method']
                cer = float(row['cer'])

                if page_num not in page_scores or cer < page_scores[page_num]['cer']:
                    page_scores[page_num] = {
                        'method': method,
                        'cer': cer,
                        'khmer_coverage': float(row['khmer_coverage']),
                        'char_count': int(row['char_count'])
                    }

        logger.info(f"Loaded scores for {len(page_scores)} pages")
        return page_scores

    def load_recovered_text(self) -> List[str]:
        """Load and parse recovered.txt into page entries."""
        recovered_path = self.forensic_dir / "recovered.txt"

        if not recovered_path.exists():
            logger.error(f"No recovered.txt found at {recovered_path}")
            return []

        with open(recovered_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by entry markers
        entries = re.split(r'━━━\s*Entry\s+\d+\s*━━━', content)

        # Clean entries
        cleaned = []
        for entry in entries:
            entry = self.normalize_text(entry)
            if entry and len(entry) > 50:
                cleaned.append(entry)

        logger.info(f"Loaded {len(cleaned)} entries from recovered.txt")
        return cleaned

    def read_page_text(self, method: str, page_num: int) -> str:
        """Read corrupted text for a specific page and method."""
        page_file = self.forensic_dir / method / f"page_{page_num:04d}.txt"

        if not page_file.exists():
            return ""

        with open(page_file, 'r', encoding='utf-8') as f:
            text = f.read()

        return self.normalize_text(text)

    def is_valid_pair(
        self,
        corrupted: str,
        clean: str
    ) -> Tuple[bool, str]:
        """Validate a training pair."""
        # Check minimum length
        if len(corrupted) < 50 or len(clean) < 50:
            return False, "Text too short"

        # Check Khmer ratio for corrupted text
        corrupted_khmer = self.get_khmer_ratio(corrupted)
        if corrupted_khmer < self.min_khmer_ratio:
            return False, f"Corrupted Khmer ratio too low: {corrupted_khmer:.2%}"

        # Check Khmer ratio for clean text (should be very high)
        clean_khmer = self.get_khmer_ratio(clean)
        if clean_khmer < 0.85:
            return False, f"Clean Khmer ratio too low: {clean_khmer:.2%}"

        # Check for PUA characters in clean text (should be none)
        if self.has_pua_chars(clean):
            return False, "Clean text contains PUA characters"

        # Calculate CER
        distance = Levenshtein.distance(corrupted, clean)
        cer = distance / max(len(corrupted), len(clean))

        # Check CER bounds
        if cer < self.min_cer:
            return False, f"Texts too similar (CER: {cer:.2%}) - not corrupted enough"

        if cer > self.max_cer:
            return False, f"Texts too different (CER: {cer:.2%}) - bad match"

        return True, f"Valid pair (CER: {cer:.2%})"

    def extract_pairs(self) -> List[Dict]:
        """Extract all training pairs from forensic directory."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing forensic directory: {self.forensic_dir.name}")
        logger.info(f"{'='*60}")

        # Load page scores
        page_scores = self.load_page_scores()

        # Load recovered text
        recovered_entries = self.load_recovered_text()

        if not recovered_entries:
            logger.error("No recovered text available")
            return []

        pairs = []
        total_cer = 0.0

        # Process each page
        for page_num in tqdm(sorted(page_scores.keys()), desc="Extracting pairs"):
            self.stats["total_pages_processed"] += 1

            # Get best method and read corrupted text
            best_method = page_scores[page_num]['method']
            corrupted_text = self.read_page_text(best_method, page_num)

            if not corrupted_text:
                self._reject_pair("No corrupted text extracted")
                continue

            # Get corresponding clean text
            # Assume page_num corresponds to entry index (1-based to 0-based)
            entry_idx = page_num - 1

            if entry_idx >= len(recovered_entries):
                self._reject_pair("No corresponding recovered entry")
                continue

            clean_text = recovered_entries[entry_idx]

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
                        "source": self.forensic_dir.name,
                        "page": page_num,
                        "method": best_method,
                        "cer": cer,
                        "input_length": len(corrupted_text),
                        "target_length": len(clean_text),
                        "input_khmer_ratio": self.get_khmer_ratio(corrupted_text),
                        "target_khmer_ratio": self.get_khmer_ratio(clean_text)
                    }
                })

                self.stats["pairs_created"] += 1
                total_cer += cer

                logger.debug(
                    f"Page {page_num}: Valid pair (CER: {cer:.2%}, method: {best_method})"
                )
            else:
                self._reject_pair(reason)
                logger.debug(f"Page {page_num}: Rejected - {reason}")

        # Calculate average CER
        if self.stats["pairs_created"] > 0:
            self.stats["avg_cer"] = total_cer / self.stats["pairs_created"]

        logger.info(f"\nResults for {self.forensic_dir.name}:")
        logger.info(f"  Pages processed: {self.stats['total_pages_processed']}")
        logger.info(f"  Pairs created: {self.stats['pairs_created']}")
        logger.info(f"  Pairs rejected: {self.stats['pairs_rejected']}")
        logger.info(f"  Average CER: {self.stats['avg_cer']:.2%}")

        return pairs

    def _reject_pair(self, reason: str):
        """Record pair rejection."""
        self.stats["pairs_rejected"] += 1
        self.stats["rejection_reasons"][reason] = \
            self.stats["rejection_reasons"].get(reason, 0) + 1

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
        logger.info("Starting forensic pair extraction")
        logger.info(f"Forensic directory: {self.forensic_dir}")
        logger.info(f"Output directory: {self.output_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract pairs
        pairs = self.extract_pairs()

        if not pairs:
            logger.error("No pairs extracted!")
            return

        # Split dataset
        train, val, test = self.split_dataset(pairs)

        logger.info(f"\nDataset split:")
        logger.info(f"  Train: {len(train)} pairs")
        logger.info(f"  Val:   {len(val)} pairs")
        logger.info(f"  Test:  {len(test)} pairs")

        # Save datasets
        self.save_dataset(train, "train")
        self.save_dataset(val, "val")
        self.save_dataset(test, "test")

        # Save statistics
        stats_file = self.output_dir / "extraction_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        logger.info(f"\nStatistics saved to {stats_file}")
        logger.info(f"Overall average CER: {self.stats['avg_cer']:.2%}")

        # Print rejection reasons
        if self.stats["rejection_reasons"]:
            logger.info("\nRejection reasons:")
            for reason, count in sorted(
                self.stats["rejection_reasons"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                logger.info(f"  {reason}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract training pairs from forensic recovery directories"
    )
    parser.add_argument(
        "--forensic-dir",
        type=Path,
        required=True,
        help="Forensic directory (e.g., freq_final, iter1_final)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for training pairs"
    )
    parser.add_argument(
        "--min-khmer-ratio",
        type=float,
        default=0.70,
        help="Minimum Khmer character ratio (0.0-1.0)"
    )
    parser.add_argument(
        "--max-cer",
        type=float,
        default=0.95,
        help="Maximum CER threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--min-cer",
        type=float,
        default=0.05,
        help="Minimum CER threshold (0.0-1.0)"
    )

    args = parser.parse_args()

    extractor = ForensicPairExtractor(
        forensic_dir=args.forensic_dir,
        output_dir=args.output_dir,
        min_khmer_ratio=args.min_khmer_ratio,
        max_cer=args.max_cer,
        min_cer=args.min_cer
    )

    extractor.run()


if __name__ == "__main__":
    main()
