#!/usr/bin/env python3
"""
Dataset Validator for Enhanced Metadata Format

Validates dataset structure, checks metadata completeness, and generates quality reports.

Usage:
    python3 validate_dataset.py --dataset-dir training_pairs_combined_257p_enhanced
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

from schema import validate_training_pair


class DatasetValidator:
    """Validate enhanced metadata datasets."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_pairs': 0,
            'valid_pairs': 0,
            'invalid_pairs': 0,
            'truncated_pairs': 0,
            'noise_patterns': Counter(),
            'fonts': Counter(),
            'extraction_methods': Counter(),
            'splits': Counter()
        }

    def validate_split(self, split: str) -> List[Dict]:
        """Validate a specific split."""
        split_file = self.dataset_dir / f"{split}.json"

        if not split_file.exists():
            self.warnings.append(f"Split file not found: {split}.json")
            return []

        with open(split_file, 'r') as f:
            pairs = json.load(f)

        print(f"\nValidating {split} split ({len(pairs)} pairs)...")

        for i, pair in enumerate(pairs):
            is_valid, errors = validate_training_pair(pair)

            if is_valid:
                self.stats['valid_pairs'] += 1

                # Collect statistics
                metadata = pair.get('metadata', {})

                if metadata.get('is_truncated'):
                    self.stats['truncated_pairs'] += 1

                for tag in metadata.get('noise_tags', []):
                    self.stats['noise_patterns'][tag] += 1

                for font in metadata.get('font_names', []):
                    self.stats['fonts'][font] += 1

                method = metadata.get('extraction')
                if method:
                    self.stats['extraction_methods'][method] += 1

                self.stats['splits'][split] += 1
            else:
                self.stats['invalid_pairs'] += 1
                self.errors.append(f"{split}[{i}]: {', '.join(errors)}")

            self.stats['total_pairs'] += 1

        return pairs

    def generate_report(self):
        """Generate validation report."""
        print("\n" + "=" * 80)
        print("📊 DATASET VALIDATION REPORT")
        print("=" * 80)

        print(f"\nDataset: {self.dataset_dir.name}")
        print(f"Total pairs: {self.stats['total_pairs']}")
        print(f"Valid pairs: {self.stats['valid_pairs']} ({self.stats['valid_pairs']/max(self.stats['total_pairs'],1)*100:.1f}%)")
        print(f"Invalid pairs: {self.stats['invalid_pairs']}")

        if self.stats['truncated_pairs'] > 0:
            print(f"\n⚠️  TRUNCATION DETECTED:")
            print(f"  Truncated pairs: {self.stats['truncated_pairs']} ({self.stats['truncated_pairs']/max(self.stats['total_pairs'],1)*100:.1f}%)")

        print(f"\n📈 Split Distribution:")
        for split, count in self.stats['splits'].items():
            print(f"  {split}: {count}")

        if self.stats['noise_patterns']:
            print(f"\n🔍 Top Noise Patterns:")
            for pattern, count in self.stats['noise_patterns'].most_common(10):
                print(f"  {pattern}: {count}")

        if self.stats['fonts']:
            print(f"\n🔤 Font Distribution:")
            for font, count in self.stats['fonts'].most_common(10):
                print(f"  {font}: {count}")

        if self.stats['extraction_methods']:
            print(f"\n⚙️  Extraction Methods:")
            for method, count in self.stats['extraction_methods'].items():
                print(f"  {method}: {count}")

        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                print(f"  {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")

        # Save report to file
        report = {
            'dataset': str(self.dataset_dir),
            'total_pairs': self.stats['total_pairs'],
            'valid_pairs': self.stats['valid_pairs'],
            'invalid_pairs': self.stats['invalid_pairs'],
            'truncated_pairs': self.stats['truncated_pairs'],
            'truncation_rate': f"{self.stats['truncated_pairs']/max(self.stats['total_pairs'],1)*100:.1f}%",
            'splits': dict(self.stats['splits']),
            'noise_patterns': dict(self.stats['noise_patterns'].most_common()),
            'fonts': dict(self.stats['fonts'].most_common()),
            'extraction_methods': dict(self.stats['extraction_methods']),
            'errors': self.errors,
            'warnings': self.warnings
        }

        report_file = self.dataset_dir / 'validation_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Report saved to: {report_file}")

        if self.stats['invalid_pairs'] == 0:
            print("\n✅ Dataset validation PASSED!")
        else:
            print(f"\n❌ Dataset validation FAILED ({self.stats['invalid_pairs']} invalid pairs)")

        return self.stats['invalid_pairs'] == 0

    def validate(self) -> bool:
        """Validate entire dataset."""
        print(f"\n🔍 Validating dataset: {self.dataset_dir}")

        self.validate_split('train')
        self.validate_split('val')
        self.validate_split('test')

        return self.generate_report()


def main():
    parser = argparse.ArgumentParser(description="Validate enhanced metadata dataset")
    parser.add_argument('--dataset-dir', type=Path, required=True, help='Dataset directory to validate')

    args = parser.parse_args()

    validator = DatasetValidator(args.dataset_dir)
    success = validator.validate()

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
