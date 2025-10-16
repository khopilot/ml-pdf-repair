#!/usr/bin/env python3
"""
Combine multiple training datasets into one mega dataset.

This script merges JSON training files from different sources while:
- Removing duplicates (by text hash)
- Preserving metadata
- Maintaining train/val/test splits
"""

import argparse
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCombiner:
    """Combine multiple training datasets."""

    def __init__(self, dataset_dirs: List[Path], output_dir: Path):
        self.dataset_dirs = [Path(d) for d in dataset_dirs]
        self.output_dir = Path(output_dir)
        self.stats = {
            "total_pairs": 0,
            "duplicates_removed": 0,
            "by_source": {}
        }

    @staticmethod
    def get_pair_hash(pair: Dict) -> str:
        """Generate unique hash for a training pair."""
        # Use input+target text to identify duplicates
        # Handle both formats: "input"/"target" and "input_text"/"target_text"
        input_text = pair.get("input") or pair.get("input_text", "")
        target_text = pair.get("target") or pair.get("target_text", "")
        text = input_text + "|" + target_text
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @staticmethod
    def normalize_pair_format(pair: Dict) -> Dict:
        """Normalize pair to use 'input' and 'target' keys."""
        # If using old format (input_text/target_text), convert to new format
        if "input_text" in pair:
            return {
                "input": pair["input_text"],
                "target": pair["target_text"],
                "metadata": {
                    "page_num": pair.get("page_num"),
                    "pdf_name": pair.get("pdf_name"),
                    "method": pair.get("method"),
                    "cer": pair.get("cer"),
                    "input_length": pair.get("input_length"),
                    "target_length": pair.get("target_length")
                }
            }
        # Already in new format
        return pair

    def load_dataset(self, dataset_dir: Path, split: str) -> List[Dict]:
        """Load a dataset split (train/val/test)."""
        split_file = dataset_dir / f"{split}.json"

        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            return []

        with open(split_file, 'r', encoding='utf-8') as f:
            pairs = json.load(f)

        logger.info(f"Loaded {len(pairs)} pairs from {dataset_dir.name}/{split}.json")
        return pairs

    def deduplicate_pairs(self, pairs: List[Dict]) -> List[Dict]:
        """Remove duplicate pairs based on text hash."""
        seen_hashes = set()
        unique_pairs = []
        duplicates = 0

        for pair in pairs:
            # Normalize format first
            normalized_pair = self.normalize_pair_format(pair)
            pair_hash = self.get_pair_hash(normalized_pair)

            if pair_hash not in seen_hashes:
                seen_hashes.add(pair_hash)
                unique_pairs.append(normalized_pair)
            else:
                duplicates += 1

        self.stats["duplicates_removed"] += duplicates
        logger.info(f"Removed {duplicates} duplicates, kept {len(unique_pairs)} unique pairs")

        return unique_pairs

    def combine_split(self, split: str) -> List[Dict]:
        """Combine a specific split across all datasets."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Combining {split} split")
        logger.info(f"{'='*60}")

        all_pairs = []

        # Load from each dataset
        for dataset_dir in self.dataset_dirs:
            pairs = self.load_dataset(dataset_dir, split)

            # Add source metadata
            for pair in pairs:
                if "metadata" not in pair:
                    pair["metadata"] = {}
                pair["metadata"]["original_source"] = dataset_dir.name

            all_pairs.extend(pairs)

            # Track stats
            source_name = dataset_dir.name
            if source_name not in self.stats["by_source"]:
                self.stats["by_source"][source_name] = {
                    "train": 0, "val": 0, "test": 0
                }
            self.stats["by_source"][source_name][split] = len(pairs)

        logger.info(f"Total pairs before deduplication: {len(all_pairs)}")

        # Deduplicate
        unique_pairs = self.deduplicate_pairs(all_pairs)

        return unique_pairs

    def save_dataset(self, pairs: List[Dict], split: str):
        """Save combined dataset to JSON file."""
        output_file = self.output_dir / f"{split}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(pairs)} pairs to {output_file}")

    def run(self):
        """Main combination pipeline."""
        logger.info("Starting dataset combination")
        logger.info(f"Input datasets:")
        for d in self.dataset_dirs:
            logger.info(f"  - {d}")
        logger.info(f"Output directory: {self.output_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Combine each split
        train_pairs = self.combine_split("train")
        val_pairs = self.combine_split("val")
        test_pairs = self.combine_split("test")

        # Save combined datasets
        self.save_dataset(train_pairs, "train")
        self.save_dataset(val_pairs, "val")
        self.save_dataset(test_pairs, "test")

        # Calculate totals
        total_pairs = len(train_pairs) + len(val_pairs) + len(test_pairs)
        self.stats["total_pairs"] = total_pairs

        # Save statistics
        stats_file = self.output_dir / "combination_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("COMBINATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total unique pairs: {total_pairs}")
        logger.info(f"  Train: {len(train_pairs)}")
        logger.info(f"  Val:   {len(val_pairs)}")
        logger.info(f"  Test:  {len(test_pairs)}")
        logger.info(f"Duplicates removed: {self.stats['duplicates_removed']}")
        logger.info(f"\nStatistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple training datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        type=Path,
        required=True,
        help="List of dataset directories to combine"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for combined dataset"
    )

    args = parser.parse_args()

    combiner = DatasetCombiner(
        dataset_dirs=args.datasets,
        output_dir=args.output_dir
    )

    combiner.run()


if __name__ == "__main__":
    main()
