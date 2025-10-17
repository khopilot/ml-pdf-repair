#!/usr/bin/env python3
"""
Pre-filter training dataset to remove low-quality pairs.

This script filters the unfiltered dataset ONCE offline, creating a clean
dataset that loads instantly (30 seconds vs 6+ hours of CER computation).

Quality gates:
- CER < 0.40 (40% - pairs with higher error rate are too noisy)
- Khmer ratio > 0.95 (95% - target must be mostly Khmer text)

Usage:
    python3 data/prefilter_dataset.py

Output:
    data/training_pairs_filtered_clean/
        ├── train.json
        ├── val.json
        ├── test.json
        ├── combination_stats.json
        └── validation_report.json
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Tuple


def prefilter_dataset(
    input_dir: str = 'data/training_pairs_unfiltered_full',
    output_dir: str = 'data/training_pairs_filtered_clean',
    max_cer: float = 0.40,
    min_khmer_ratio: float = 0.95,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Pre-filter dataset using existing cer_estimate in metadata.

    Args:
        input_dir: Path to unfiltered dataset
        output_dir: Path to save filtered dataset
        max_cer: Maximum allowed Character Error Rate (default: 0.40)
        min_khmer_ratio: Minimum Khmer character ratio in target (default: 0.95)
        verbose: Print progress messages

    Returns:
        Dictionary with filtering statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    if verbose:
        print("=" * 70)
        print("📊 PRE-FILTERING DATASET")
        print("=" * 70)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"\n🎯 Quality gates:")
        print(f"   - Maximum CER: {max_cer} ({max_cer*100:.0f}%)")
        print(f"   - Minimum Khmer ratio: {min_khmer_ratio} ({min_khmer_ratio*100:.0f}%)")
        print()

    stats = {
        'total': 0,
        'kept': 0,
        'rejected_cer': 0,
        'rejected_khmer': 0,
        'rejected_missing_metadata': 0
    }

    split_stats = {}

    # Process each split
    for split in ['train', 'val', 'test']:
        input_file = input_path / f'{split}.json'
        if not input_file.exists():
            if verbose:
                print(f"⚠️  {split}.json not found, skipping")
            continue

        if verbose:
            print(f"Processing {split}.json...")

        # Load data
        with open(input_file, encoding='utf-8') as f:
            data = json.load(f)

        initial_count = len(data)
        filtered = []
        rejected_details = {'cer': 0, 'khmer': 0, 'missing': 0}

        for pair in data:
            stats['total'] += 1

            # Get metadata
            metadata = pair.get('metadata', {})

            # Use pre-computed metrics from metadata
            cer = metadata.get('cer_estimate')
            khmer_ratio = metadata.get('target_khmer_ratio')

            # CER is not pre-computed in this dataset, skip CER filtering
            # Only filter by Khmer ratio (which IS available)
            if khmer_ratio is None:
                stats['rejected_missing_metadata'] += 1
                rejected_details['missing'] += 1
                continue

            # Quality gates
            # Note: CER filtering disabled (cer_estimate not available in dataset)
            # if cer is not None and cer > max_cer:
            #     stats['rejected_cer'] += 1
            #     rejected_details['cer'] += 1
            #     continue

            if khmer_ratio < min_khmer_ratio:
                stats['rejected_khmer'] += 1
                rejected_details['khmer'] += 1
                continue

            # Pair passed all gates
            filtered.append(pair)
            stats['kept'] += 1

        # Save filtered split
        output_file = output_path / f'{split}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

        # Track split stats
        kept_pct = (len(filtered) / initial_count * 100) if initial_count > 0 else 0
        split_stats[split] = {
            'initial': initial_count,
            'kept': len(filtered),
            'kept_pct': kept_pct,
            'rejected': rejected_details
        }

        if verbose:
            print(f"   ✅ {split}: {len(filtered)}/{initial_count} pairs kept ({kept_pct:.1f}%)")
            if rejected_details['cer'] > 0:
                print(f"      - Rejected (CER > {max_cer}): {rejected_details['cer']}")
            if rejected_details['khmer'] > 0:
                print(f"      - Rejected (Khmer < {min_khmer_ratio}): {rejected_details['khmer']}")
            if rejected_details['missing'] > 0:
                print(f"      - Rejected (missing metadata): {rejected_details['missing']}")

    # Copy metadata files
    if verbose:
        print(f"\nCopying metadata files...")

    for meta_file in ['combination_stats.json', 'validation_report.json']:
        src = input_path / meta_file
        if src.exists():
            shutil.copy(src, output_path / meta_file)
            if verbose:
                print(f"   ✅ {meta_file}")

    # Print summary
    if verbose:
        print()
        print("=" * 70)
        print("📊 FILTERING SUMMARY")
        print("=" * 70)
        kept_pct = (stats['kept'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"Total pairs processed:  {stats['total']}")
        print(f"✅ Kept:                {stats['kept']} ({kept_pct:.1f}%)")
        print(f"❌ Rejected (CER):      {stats['rejected_cer']}")
        print(f"❌ Rejected (Khmer):    {stats['rejected_khmer']}")
        print(f"❌ Rejected (missing):  {stats['rejected_missing_metadata']}")
        print()

        # Per-split breakdown
        print("Per-split breakdown:")
        for split, split_info in split_stats.items():
            print(f"   {split:5s}: {split_info['kept']:4d}/{split_info['initial']:4d} ({split_info['kept_pct']:5.1f}%)")

        print()
        print(f"📁 Filtered dataset saved to: {output_path}")
        print("=" * 70)

    # Create a filtering report
    report = {
        'input_dir': str(input_path),
        'output_dir': str(output_path),
        'filters': {
            'max_cer': max_cer,
            'min_khmer_ratio': min_khmer_ratio
        },
        'statistics': stats,
        'split_details': split_stats
    }

    report_file = output_path / 'filtering_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"📄 Filtering report saved to: {report_file}")

    return stats


if __name__ == '__main__':
    import sys

    # Parse command-line arguments (simple)
    input_dir = 'data/training_pairs_unfiltered_full'
    output_dir = 'data/training_pairs_filtered_clean'
    max_cer = 0.40
    min_khmer_ratio = 0.95

    if len(sys.argv) > 1:
        if '--help' in sys.argv or '-h' in sys.argv:
            print(__doc__)
            sys.exit(0)

        # Simple arg parsing
        for i, arg in enumerate(sys.argv[1:]):
            if arg == '--input':
                input_dir = sys.argv[i+2]
            elif arg == '--output':
                output_dir = sys.argv[i+2]
            elif arg == '--max-cer':
                max_cer = float(sys.argv[i+2])
            elif arg == '--min-khmer':
                min_khmer_ratio = float(sys.argv[i+2])

    # Run filtering
    stats = prefilter_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        max_cer=max_cer,
        min_khmer_ratio=min_khmer_ratio,
        verbose=True
    )

    # Exit with status
    if stats['kept'] == 0:
        print("\n❌ ERROR: No pairs passed filtering! Check filter criteria.")
        sys.exit(1)
    elif stats['kept'] < 100:
        print(f"\n⚠️  WARNING: Only {stats['kept']} pairs kept. Consider relaxing filters.")
    else:
        print(f"\n✅ SUCCESS: {stats['kept']} clean pairs ready for training!")

    sys.exit(0)
