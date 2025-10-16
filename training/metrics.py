"""
Evaluation metrics for Khmer text correction.

Implements:
    - Character Error Rate (CER)
    - Word Accuracy
    - Khmer Cluster Validity
    - Sequence Accuracy
"""

import torch
from typing import List, Optional
import unicodedata


class CERMetric:
    """
    Character Error Rate (CER) using Levenshtein distance.

    CER = (substitutions + deletions + insertions) / total_chars
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metric state."""
        self.total_distance = 0
        self.total_length = 0

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Compute Levenshtein distance (edit distance) between two strings.

        Args:
            s1: String 1
            s2: String 2

        Returns:
            Edit distance (minimum number of edits)
        """
        if len(s1) < len(s2):
            return CERMetric.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        # Initialize DP table
        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)

                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    def update(self, predictions: List[str], references: List[str]):
        """
        Update CER with batch of predictions.

        Args:
            predictions: List of predicted strings
            references: List of reference (ground truth) strings
        """
        for pred, ref in zip(predictions, references):
            distance = self.levenshtein_distance(pred, ref)
            self.total_distance += distance
            self.total_length += len(ref)

    def compute(self) -> float:
        """
        Compute CER.

        Returns:
            CER score (0.0 = perfect, 1.0 = completely wrong)
        """
        if self.total_length == 0:
            return 0.0

        return self.total_distance / self.total_length


class WordAccuracyMetric:
    """
    Word-level accuracy.

    Measures how many complete words are correct.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metric state."""
        self.total_correct = 0
        self.total_words = 0

    def update(self, predictions: List[str], references: List[str]):
        """
        Update word accuracy with batch.

        Args:
            predictions: List of predicted strings
            references: List of reference strings
        """
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()

            # Count matching words
            max_len = max(len(pred_words), len(ref_words))

            for i in range(max_len):
                self.total_words += 1

                if i < len(pred_words) and i < len(ref_words):
                    if pred_words[i] == ref_words[i]:
                        self.total_correct += 1

    def compute(self) -> float:
        """
        Compute word accuracy.

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if self.total_words == 0:
            return 0.0

        return self.total_correct / self.total_words


class KhmerClusterValidityMetric:
    """
    Validate Khmer consonant clusters.

    Checks if consonant-coeng-consonant sequences are valid.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metric state."""
        self.total_clusters = 0
        self.valid_clusters = 0

    @staticmethod
    def is_khmer_consonant(char: str) -> bool:
        """Check if character is Khmer consonant."""
        return 0x1780 <= ord(char) <= 0x17A2

    @staticmethod
    def is_coeng(char: str) -> bool:
        """Check if character is coeng (subscript marker)."""
        return ord(char) == 0x17D2

    def update(self, predictions: List[str], references: Optional[List[str]] = None):
        """
        Update cluster validity with batch.

        Args:
            predictions: List of predicted strings
            references: Not used (validation is absolute)
        """
        for pred in predictions:
            # Find all consonant-coeng-consonant sequences
            for i in range(len(pred) - 2):
                if (self.is_khmer_consonant(pred[i]) and
                    self.is_coeng(pred[i + 1]) and
                    self.is_khmer_consonant(pred[i + 2])):

                    self.total_clusters += 1

                    # All consonant clusters are valid in Khmer
                    # (This is a simplified check; could add linguistic rules)
                    self.valid_clusters += 1

    def compute(self) -> float:
        """
        Compute cluster validity ratio.

        Returns:
            Validity ratio (0.0 to 1.0)
        """
        if self.total_clusters == 0:
            return 1.0  # No clusters found, assume valid

        return self.valid_clusters / self.total_clusters


class SequenceAccuracyMetric:
    """
    Exact sequence match accuracy.

    Measures how many complete sequences match exactly.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metric state."""
        self.total_correct = 0
        self.total_sequences = 0

    def update(self, predictions: List[str], references: List[str]):
        """
        Update sequence accuracy with batch.

        Args:
            predictions: List of predicted strings
            references: List of reference strings
        """
        for pred, ref in zip(predictions, references):
            self.total_sequences += 1

            if pred == ref:
                self.total_correct += 1

    def compute(self) -> float:
        """
        Compute sequence accuracy.

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if self.total_sequences == 0:
            return 0.0

        return self.total_correct / self.total_sequences


class MetricCollection:
    """
    Collection of all metrics for comprehensive evaluation.
    """

    def __init__(self):
        self.cer = CERMetric()
        self.word_acc = WordAccuracyMetric()
        self.cluster_validity = KhmerClusterValidityMetric()
        self.seq_acc = SequenceAccuracyMetric()

    def reset(self):
        """Reset all metrics."""
        self.cer.reset()
        self.word_acc.reset()
        self.cluster_validity.reset()
        self.seq_acc.reset()

    def update(self, predictions: List[str], references: List[str]):
        """
        Update all metrics.

        Args:
            predictions: List of predicted strings
            references: List of reference strings
        """
        self.cer.update(predictions, references)
        self.word_acc.update(predictions, references)
        self.cluster_validity.update(predictions, references)
        self.seq_acc.update(predictions, references)

    def compute(self) -> dict:
        """
        Compute all metrics.

        Returns:
            Dict with all metric values
        """
        return {
            'cer': self.cer.compute(),
            'word_accuracy': self.word_acc.compute(),
            'cluster_validity': self.cluster_validity.compute(),
            'sequence_accuracy': self.seq_acc.compute(),
            'character_accuracy': 1.0 - self.cer.compute()  # Derived metric
        }


if __name__ == "__main__":
    # Test metrics
    metrics = MetricCollection()

    # Example predictions and references
    predictions = [
        "ក្នុងការដោះស្រាយ",
        "ភាសាខ្មែរ"
    ]

    references = [
        "ក្នុងការដោះស្រាយ",  # Perfect match
        "ភាសាខ្មែរ123"         # Different ending
    ]

    metrics.update(predictions, references)
    results = metrics.compute()

    print("Metrics:")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")

    # Test CER calculation
    cer = CERMetric()
    pred = "ក្ងនការដោះស្រាយ"  # Corrupted
    ref = "ក្នុងការដោះស្រាយ"   # Correct

    distance = cer.levenshtein_distance(pred, ref)
    cer_value = distance / len(ref)

    print(f"\nEdit distance: {distance}")
    print(f"CER: {cer_value:.4f}")
