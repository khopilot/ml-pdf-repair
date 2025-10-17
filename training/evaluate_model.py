#!/usr/bin/env python3
"""
Evaluate trained Khmer text correction model on test set.

Supports:
- Loading checkpoints (HybridCorrector, Transformer, LSTM)
- Computing metrics (CER, character accuracy, Khmer coverage)
- Generating prediction samples
- Exporting results to JSON and Markdown

Usage:
    python evaluate_model.py \\
        --checkpoint ../trained_models/ams_2558p_50epoch/best_model.pt \\
        --data-dir data/training_pairs_mega_331p \\
        --output-dir evaluation_results/ams_2558p_on_mega331p \\
        --device mps \\
        --num-samples 10
"""

import argparse
import json
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import Levenshtein

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import KhmerCharVocab, KhmerCorrectionDataset
from models.hybrid_corrector import HybridCorrector
from models.char_transformer import KhmerTransformerCorrector
from models.char_lstm import CharLSTMSeq2Seq


class ModelEvaluator:
    """
    Evaluator for Khmer text correction models.

    Loads trained checkpoints and evaluates on test data.
    """

    def __init__(self,
                 checkpoint_path: Path,
                 vocab: KhmerCharVocab,
                 device: torch.device,
                 model_type: Optional[str] = None):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            vocab: Character vocabulary
            device: Torch device
            model_type: Model architecture (hybrid, transformer, lstm) or None to auto-detect
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab = vocab
        self.device = device

        # Load checkpoint
        print(f"Loading checkpoint from {self.checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract metadata
        self.epoch = self.checkpoint.get('epoch', 'unknown')
        self.global_step = self.checkpoint.get('global_step', 'unknown')
        self.checkpoint_metrics = self.checkpoint.get('metrics', {})

        print(f"Checkpoint info:")
        print(f"  Epoch: {self.epoch}")
        print(f"  Global step: {self.global_step}")
        if self.checkpoint_metrics:
            print(f"  Val loss: {self.checkpoint_metrics.get('loss', 'N/A')}")
            print(f"  Val CER: {self.checkpoint_metrics.get('cer', 'N/A')}")

        # Detect or use specified model type
        self.model_type = model_type or self._detect_model_type()
        print(f"Model type: {self.model_type}")

        # Load model
        self.model = self._load_model()
        self.model.to(device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _detect_model_type(self) -> str:
        """Detect model type from checkpoint structure."""
        state_dict = self.checkpoint['model_state_dict']

        # Check for hybrid-specific keys
        if any('atomic_mapper' in k for k in state_dict.keys()):
            return 'hybrid'

        # Check for LSTM-specific keys
        if any('lstm' in k.lower() for k in state_dict.keys()):
            return 'lstm'

        # Default to transformer
        return 'transformer'

    def _load_model(self) -> nn.Module:
        """Load model architecture and weights from checkpoint."""
        vocab_size = self.checkpoint['vocab_size']

        if self.model_type == 'hybrid':
            # HybridCorrector architecture
            model = HybridCorrector(
                vocab_size=vocab_size,
                atomic_embed_dim=128,
                atomic_hidden_dim=256,
                atomic_num_layers=3,
                refiner_d_model=128,
                refiner_nhead=4,
                refiner_num_encoder_layers=3,
                refiner_num_decoder_layers=3,
                refiner_dim_feedforward=512,
                dropout=0.1,
                pad_token_id=self.vocab.pad_token_id
            )

        elif self.model_type == 'transformer':
            # KhmerTransformerCorrector architecture
            model = KhmerTransformerCorrector(
                vocab_size=vocab_size,
                d_model=256,
                nhead=8,
                num_encoder_layers=4,
                num_decoder_layers=4,
                dim_feedforward=1024,
                dropout=0.1,
                pad_token_id=self.vocab.pad_token_id
            )

        elif self.model_type == 'lstm':
            # CharLSTMSeq2Seq architecture
            model = CharLSTMSeq2Seq(
                vocab_size=vocab_size,
                embed_dim=128,
                encoder_hidden_dim=256,
                decoder_hidden_dim=256,
                num_layers=2,
                dropout=0.1,
                pad_token_id=self.vocab.pad_token_id
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Load weights
        model.load_state_dict(self.checkpoint['model_state_dict'])

        return model

    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_data: List of test examples (input/target dicts)

        Returns:
            Dict of metrics (cer, character_accuracy, etc.)
        """
        print(f"\nEvaluating on {len(test_data)} test samples...")

        total_cer = 0.0
        total_chars = 0
        correct_chars = 0
        correct_sequences = 0
        total_khmer_output = 0
        total_khmer_target = 0
        inference_times = []

        predictions = []

        for example in tqdm(test_data, desc="Evaluating"):
            input_text = example['input']
            target_text = example['target']

            # Inference
            start_time = time.time()
            pred_text = self.predict(input_text)
            inference_time = (time.time() - start_time) * 1000  # ms
            inference_times.append(inference_time)

            # Store prediction
            predictions.append({
                'input': input_text,
                'target': target_text,
                'prediction': pred_text
            })

            # Compute CER
            distance = Levenshtein.distance(pred_text, target_text)
            max_len = max(len(pred_text), len(target_text))
            cer = distance / max_len if max_len > 0 else 0.0
            total_cer += cer

            # Character accuracy
            for pred_char, target_char in zip(pred_text, target_text):
                if pred_char == target_char:
                    correct_chars += 1
            total_chars += max_len

            # Sequence accuracy
            if pred_text == target_text:
                correct_sequences += 1

            # Khmer coverage
            total_khmer_output += sum(1 for c in pred_text if self._is_khmer(c))
            total_khmer_target += sum(1 for c in target_text if self._is_khmer(c))

        # Compute metrics
        metrics = {
            'cer': total_cer / len(test_data),
            'character_accuracy': correct_chars / total_chars if total_chars > 0 else 0.0,
            'sequence_accuracy': correct_sequences / len(test_data),
            'khmer_coverage_output': total_khmer_output / len(''.join(p['prediction'] for p in predictions)) if predictions else 0.0,
            'khmer_coverage_target': total_khmer_target / len(''.join(p['target'] for p in predictions)) if predictions else 0.0,
            'avg_inference_time_ms': sum(inference_times) / len(inference_times) if inference_times else 0.0,
            'total_samples': len(test_data)
        }

        return metrics, predictions

    def predict(self, input_text: str, max_length: int = 512, repetition_penalty: float = 1.2) -> str:
        """
        Generate prediction for input text with repetition guards.

        Args:
            input_text: Corrupted input text
            max_length: Maximum output length
            repetition_penalty: Penalty factor for repeated tokens (default: 1.2)

        Returns:
            Corrected text prediction
        """
        # Encode input
        input_ids = self.vocab.encode(input_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Create target with START token
        BOS = self.vocab.start_token_id
        EOS = self.vocab.end_token_id
        y = torch.tensor([[BOS]], dtype=torch.long, device=self.device)

        # Padding masks
        src_padding_mask = (input_ids == self.vocab.pad_token_id)

        with torch.no_grad():
            # Generate autoregressively with safety guards
            for step in range(max_length):
                tgt_padding_mask = (y == self.vocab.pad_token_id)

                # Forward pass
                if self.model_type == 'hybrid':
                    outputs = self.model(
                        input_ids,
                        y,
                        src_padding_mask,
                        tgt_padding_mask
                    )
                elif self.model_type == 'transformer':
                    outputs = self.model(
                        input_ids,
                        y,
                        src_padding_mask,
                        tgt_padding_mask
                    )
                elif self.model_type == 'lstm':
                    input_lengths = [input_ids.size(1)]
                    target_lengths = [y.size(1)]
                    outputs = self.model(
                        input_ids,
                        y,
                        input_lengths,
                        target_lengths,
                        teacher_forcing_ratio=0.0
                    )

                # Get next token logits
                next_logits = outputs[0, -1, :].clone()  # [vocab_size]

                # SAFETY GUARD 1: Repetition penalty on last 20 tokens
                if y.size(1) > 1:
                    for tok in y[0, -min(20, y.size(1)):].tolist():
                        if tok not in [self.vocab.pad_token_id, BOS, EOS]:
                            next_logits[tok] /= repetition_penalty

                # Get top-2 candidates
                top2_logits, top2_indices = next_logits.topk(2)
                next_token = top2_indices[0].item()

                # SAFETY GUARD 2: Avoid triple-repeat of same character
                if y.size(1) >= 2:
                    if next_token == y[0, -1].item() == y[0, -2].item():
                        # Use second-best token instead
                        next_token = top2_indices[1].item()

                # SAFETY GUARD 3: Stop on EOS
                if EOS is not None and next_token == EOS:
                    break

                # SAFETY GUARD 4: Stop if generating only padding
                if next_token == self.vocab.pad_token_id:
                    break

                # Append to target
                y = torch.cat([
                    y,
                    torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                ], dim=1)

        # Decode prediction (skip BOS token)
        pred_ids = y[0, 1:].tolist()
        pred_text = self.vocab.decode(pred_ids, skip_special_tokens=True)

        return pred_text

    @staticmethod
    def _is_khmer(char: str) -> bool:
        """Check if character is Khmer Unicode."""
        return '\u1780' <= char <= '\u17FF'

    def generate_samples(self, predictions: List[Dict], num_samples: int = 10) -> List[Dict]:
        """
        Generate sample predictions for visualization.

        Args:
            predictions: All predictions
            num_samples: Number of samples to generate

        Returns:
            List of sample dicts with input/pred/target
        """
        import random

        # Sample random predictions
        samples = random.sample(predictions, min(num_samples, len(predictions)))

        # Add CER for each sample
        for sample in samples:
            pred = sample['prediction']
            target = sample['target']
            distance = Levenshtein.distance(pred, target)
            max_len = max(len(pred), len(target))
            sample['cer'] = distance / max_len if max_len > 0 else 0.0

        return samples


def save_results(metrics: Dict, predictions: List[Dict], samples: List[Dict],
                 output_dir: Path, checkpoint_info: Dict, dataset_name: str):
    """
    Save evaluation results to files.

    Args:
        metrics: Evaluation metrics
        predictions: All predictions
        samples: Sample predictions
        output_dir: Output directory
        checkpoint_info: Checkpoint metadata
        dataset_name: Name of test dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics.json
    results = {
        'dataset': dataset_name,
        'checkpoint': str(checkpoint_info['path']),
        'model_type': checkpoint_info['model_type'],
        'epoch': checkpoint_info['epoch'],
        'global_step': checkpoint_info['global_step'],
        'checkpoint_metrics': checkpoint_info['checkpoint_metrics'],
        'test_metrics': metrics
    }

    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved metrics to {output_dir / 'metrics.json'}")

    # Save full predictions
    with open(output_dir / 'predictions_full.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved full predictions to {output_dir / 'predictions_full.json'}")

    # Save samples as markdown
    samples_md = "# Prediction Samples\n\n"
    samples_md += f"**Dataset:** {dataset_name}\n"
    samples_md += f"**Checkpoint:** {checkpoint_info['path'].name}\n"
    samples_md += f"**Epoch:** {checkpoint_info['epoch']}\n"
    samples_md += f"**Test CER:** {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)\n\n"
    samples_md += "---\n\n"

    for i, sample in enumerate(samples, 1):
        cer_status = "✅ Good" if sample['cer'] < 0.3 else "⚠️ Fair" if sample['cer'] < 0.6 else "❌ Poor"

        samples_md += f"## Sample {i}\n\n"
        samples_md += f"**Input (corrupted):**\n```\n{sample['input'][:200]}{'...' if len(sample['input']) > 200 else ''}\n```\n\n"
        samples_md += f"**Prediction:**\n```\n{sample['prediction'][:200]}{'...' if len(sample['prediction']) > 200 else ''}\n```\n\n"
        samples_md += f"**Target (ground truth):**\n```\n{sample['target'][:200]}{'...' if len(sample['target']) > 200 else ''}\n```\n\n"
        samples_md += f"**CER:** {sample['cer']:.4f} ({sample['cer']*100:.2f}%) - {cer_status}\n\n"
        samples_md += "---\n\n"

    with open(output_dir / 'samples.md', 'w', encoding='utf-8') as f:
        f.write(samples_md)

    print(f"✅ Saved samples to {output_dir / 'samples.md'}")


def print_metrics_summary(metrics: Dict):
    """Print metrics summary to console."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"\nCharacter Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"Character Accuracy: {metrics['character_accuracy']:.4f} ({metrics['character_accuracy']*100:.2f}%)")
    print(f"Sequence Accuracy: {metrics['sequence_accuracy']:.4f} ({metrics['sequence_accuracy']*100:.2f}%)")
    print(f"\nKhmer Coverage (output): {metrics['khmer_coverage_output']:.4f} ({metrics['khmer_coverage_output']*100:.2f}%)")
    print(f"Khmer Coverage (target): {metrics['khmer_coverage_target']:.4f} ({metrics['khmer_coverage_target']*100:.2f}%)")
    print(f"\nAvg inference time: {metrics['avg_inference_time_ms']:.2f} ms/sample")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Khmer text correction model")

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')

    # Optional arguments
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device')
    parser.add_argument('--model-type', type=str, default=None, choices=['hybrid', 'transformer', 'lstm'], help='Model type (auto-detect if not specified)')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of sample predictions to generate')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum output length')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Check files exist
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    test_json = data_dir / 'test.json'
    if not test_json.exists():
        print(f"❌ Test data not found: {test_json}")
        return

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = KhmerCharVocab()
    print(f"Vocabulary size: {vocab.vocab_size}")

    # Load evaluator
    evaluator = ModelEvaluator(
        checkpoint_path=checkpoint_path,
        vocab=vocab,
        device=device,
        model_type=args.model_type
    )

    # Load test data
    print(f"\nLoading test data from {test_json}")
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test samples")

    # Evaluate
    metrics, predictions = evaluator.evaluate(test_data)

    # Print summary
    print_metrics_summary(metrics)

    # Generate samples
    print(f"\nGenerating {args.num_samples} sample predictions...")
    samples = evaluator.generate_samples(predictions, num_samples=args.num_samples)

    # Save results
    checkpoint_info = {
        'path': checkpoint_path,
        'model_type': evaluator.model_type,
        'epoch': evaluator.epoch,
        'global_step': evaluator.global_step,
        'checkpoint_metrics': evaluator.checkpoint_metrics
    }

    save_results(
        metrics=metrics,
        predictions=predictions,
        samples=samples,
        output_dir=output_dir,
        checkpoint_info=checkpoint_info,
        dataset_name=data_dir.name
    )

    print(f"\n🎉 Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
