"""
PyTorch Dataset classes for Khmer PDF corruption correction.

Implements character-level datasets for sequence-to-sequence training.
"""

import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import unicodedata


class KhmerCharVocab:
    """
    Character vocabulary for Khmer text.

    Covers Khmer Unicode block (U+1780-U+17FF) + special tokens.
    """

    def __init__(self):
        """Build character vocabulary."""
        self.char2idx = {}
        self.idx2char = {}

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'

        self._build_vocab()

    def _build_vocab(self):
        """Build character→index mapping."""
        idx = 0

        # Special tokens first
        for token in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]:
            self.char2idx[token] = idx
            self.idx2char[idx] = token
            idx += 1

        # Khmer Unicode block (U+1780-U+17FF)
        for cp in range(0x1780, 0x1800):
            char = chr(cp)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
            idx += 1

        # Khmer Symbols (U+19E0-U+19FF)
        for cp in range(0x19E0, 0x1A00):
            char = chr(cp)
            if char not in self.char2idx:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1

        # Common punctuation and spaces
        for char in [' ', '.', ',', '!', '?', ':', ';', '-', '(', ')', '\n']:
            if char not in self.char2idx:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1

        # Digits (both ASCII and Khmer)
        for cp in range(ord('0'), ord('9') + 1):
            char = chr(cp)
            if char not in self.char2idx:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1

        for cp in range(0x17E0, 0x17EA):  # Khmer digits
            char = chr(cp)
            if char not in self.char2idx:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.char2idx)

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        return self.char2idx[self.PAD_TOKEN]

    @property
    def start_token_id(self) -> int:
        """Get START token ID."""
        return self.char2idx[self.START_TOKEN]

    @property
    def end_token_id(self) -> int:
        """Get END token ID."""
        return self.char2idx[self.END_TOKEN]

    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self.char2idx[self.UNK_TOKEN]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Add START/END tokens

        Returns:
            List of token IDs
        """
        ids = []

        if add_special_tokens:
            ids.append(self.start_token_id)

        for char in text:
            ids.append(self.char2idx.get(char, self.unk_token_id))

        if add_special_tokens:
            ids.append(self.end_token_id)

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Skip PAD/START/END/UNK tokens

        Returns:
            Decoded text string
        """
        special_ids = {
            self.pad_token_id,
            self.start_token_id,
            self.end_token_id,
            self.unk_token_id
        } if skip_special_tokens else set()

        chars = []
        for idx in ids:
            if idx in special_ids:
                continue
            chars.append(self.idx2char.get(idx, ''))

        return ''.join(chars)


class KhmerCorrectionDataset(Dataset):
    """
    PyTorch Dataset for character-level Khmer text correction.

    Each sample: (corrupted_text_ids, correct_text_ids)
    """

    def __init__(self,
                 json_file: Path,
                 vocab: KhmerCharVocab,
                 max_length: int = 1024,
                 filter_quality: bool = True,
                 max_cer: float = 0.40,
                 min_khmer_ratio: float = 0.95):
        """
        Initialize dataset with quality filtering.

        Args:
            json_file: Path to JSON file with training pairs
            vocab: Character vocabulary
            max_length: Maximum sequence length
            filter_quality: Enable data quality filtering
            max_cer: Maximum allowed CER for a pair (default: 0.40 = 40%)
            min_khmer_ratio: Minimum Khmer character ratio in target (default: 0.95)
        """
        self.vocab = vocab
        self.max_length = max_length

        # Load data
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        initial_count = len(self.data)

        # Filter by length (handle both formats)
        self.data = [
            pair for pair in self.data
            if len(pair.get('input', pair.get('input_text', ''))) <= max_length
            and len(pair.get('target', pair.get('target_text', ''))) <= max_length
        ]

        length_filtered = initial_count - len(self.data)

        # QUALITY FILTERING: Remove garbage pairs
        if filter_quality:
            filtered_data = []
            quality_rejected = 0

            for pair in self.data:
                input_text = pair.get('input', pair.get('input_text', ''))
                target_text = pair.get('target', pair.get('target_text', ''))

                # Calculate CER if available in metadata
                cer = pair.get('cer', pair.get('initial_cer', None))
                if cer is None:
                    # Compute CER on-the-fly if not provided
                    cer = self._compute_cer(input_text, target_text)

                # Calculate Khmer ratio in target
                khmer_ratio = self._khmer_ratio(target_text)

                # Quality gates
                if cer <= max_cer and khmer_ratio >= min_khmer_ratio:
                    filtered_data.append(pair)
                else:
                    quality_rejected += 1

            self.data = filtered_data
            print(f"📊 Quality filtering: rejected {quality_rejected}/{initial_count} pairs")
            print(f"   - CER > {max_cer}: filtering high-noise pairs")
            print(f"   - Khmer ratio < {min_khmer_ratio}: filtering non-Khmer text")

        print(f"✅ Loaded {len(self.data)} training pairs from {json_file}")
        if length_filtered > 0:
            print(f"   (filtered {length_filtered} for length > {max_length})")

    @staticmethod
    def _compute_cer(text1: str, text2: str) -> float:
        """Compute Character Error Rate between two texts."""
        if len(text2) == 0:
            return 1.0 if len(text1) > 0 else 0.0

        # Simple Levenshtein distance (character-level)
        import numpy as np
        m, n = len(text1), len(text2)
        dp = np.zeros((m + 1, n + 1))

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return float(dp[m][n]) / len(text2)

    @staticmethod
    def _khmer_ratio(text: str) -> float:
        """Calculate ratio of Khmer characters in text."""
        if len(text) == 0:
            return 0.0

        khmer_count = sum(1 for char in text if '\u1780' <= char <= '\u17FF')
        return khmer_count / len(text)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dict with:
                - input_ids: Encoded corrupted text
                - target_ids: Encoded correct text
                - input_length: Actual input length (before padding)
                - target_length: Actual target length (before padding)
        """
        pair = self.data[idx]

        # Encode texts (handle both formats)
        input_text = pair.get('input', pair.get('input_text', ''))
        target_text = pair.get('target', pair.get('target_text', ''))

        input_ids = self.vocab.encode(input_text, add_special_tokens=False)
        target_ids = self.vocab.encode(target_text, add_special_tokens=True)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'input_length': len(input_ids),
            'target_length': len(target_ids),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]],
               pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads sequences to same length within batch.

    Args:
        batch: List of samples from dataset
        pad_token_id: ID for padding token

    Returns:
        Batched tensors with padding
    """
    # Extract sequences
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    input_lengths = torch.tensor([item['input_length'] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)

    # Pad sequences
    input_ids_padded = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id
    )

    target_ids_padded = pad_sequence(
        target_ids,
        batch_first=True,
        padding_value=pad_token_id
    )

    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded,
        'input_lengths': input_lengths,
        'target_lengths': target_lengths,
    }


def create_dataloaders(data_dir: Path,
                      vocab: KhmerCharVocab,
                      batch_size: int = 32,
                      max_length: int = 4000,
                      num_workers: int = 4) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train/val/test dataloaders.

    Args:
        data_dir: Directory with train.json, val.json, test.json
        vocab: Character vocabulary
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader, test_loader) tuple
    """
    from torch.utils.data import DataLoader
    from functools import partial

    # Create datasets
    train_dataset = KhmerCorrectionDataset(
        data_dir / 'train.json',
        vocab,
        max_length
    )

    val_dataset = KhmerCorrectionDataset(
        data_dir / 'val.json',
        vocab,
        max_length
    )

    test_dataset = KhmerCorrectionDataset(
        data_dir / 'test.json',
        vocab,
        max_length
    )

    # Create collate function with vocab's pad_token_id
    collate_fn_with_pad = partial(collate_fn, pad_token_id=vocab.pad_token_id)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_with_pad,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_pad,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_pad,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test vocabulary
    vocab = KhmerCharVocab()
    print(f"Vocabulary size: {vocab.vocab_size}")

    # Test encoding/decoding
    test_text = "ក្នុងការដោះស្រាយ"
    encoded = vocab.encode(test_text, add_special_tokens=True)
    decoded = vocab.decode(encoded)

    print(f"Original: {test_text}")
    print(f"Encoded: {encoded[:10]}... ({len(encoded)} tokens)")
    print(f"Decoded: {decoded}")

    # Test dataset (if data available)
    data_dir = Path(__file__).parent.parent / "data" / "training_pairs"
    if (data_dir / "train.json").exists():
        dataset = KhmerCorrectionDataset(
            data_dir / "train.json",
            vocab,
            max_length=1024
        )
        print(f"\nDataset size: {len(dataset)}")

        # Test sample
        sample = dataset[0]
        print(f"Sample input length: {sample['input_length']}")
        print(f"Sample target length: {sample['target_length']}")
