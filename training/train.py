#!/usr/bin/env python3
"""
Training script for Khmer PDF corruption correction models.

Supports both LSTM baseline and Transformer models with:
    - WandB experiment tracking
    - Automatic checkpointing
    - Early stopping
    - Learning rate scheduling
    - Mixed precision training (FP16)

Usage:
    # LSTM baseline
    python train.py \\
        --model lstm \\
        --data-dir data/training_pairs \\
        --output-dir checkpoints/lstm_baseline \\
        --epochs 50 \\
        --batch-size 64

    # Transformer (production)
    python train.py \\
        --model transformer \\
        --data-dir data/training_pairs \\
        --output-dir checkpoints/transformer_v1 \\
        --epochs 100 \\
        --batch-size 32 \\
        --wandb-project khmer-correction
"""

import argparse
import json
import logging
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import KhmerCharVocab, create_dataloaders
from models.char_lstm import CharLSTMSeq2Seq
from models.char_transformer import KhmerTransformerCorrector
from training.metrics import MetricCollection


class Trainer:
    """
    Trainer class for seq2seq models (DeepMind style).

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 vocab: KhmerCharVocab,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 device: torch.device,
                 output_dir: Path,
                 use_wandb: bool = False,
                 wandb_project: Optional[str] = None,
                 gradient_clip: float = 1.0,
                 mixed_precision: bool = False):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            vocab: Character vocabulary
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Torch device
            output_dir: Where to save checkpoints
            use_wandb: Use Weights & Biases logging
            wandb_project: WandB project name
            gradient_clip: Gradient clipping norm
            mixed_precision: Use FP16 training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.gradient_clip = gradient_clip

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.mixed_precision = mixed_precision

        # Tracking
        self.best_val_cer = float('inf')
        self.current_epoch = 0
        self.global_step = 0

        # Metrics
        self.train_metrics = MetricCollection()
        self.val_metrics = MetricCollection()

        # WandB
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, config={
                    'model': type(model).__name__,
                    'vocab_size': vocab.vocab_size,
                    'batch_size': train_loader.batch_size,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                })
                self.wandb = wandb
            except ImportError:
                logging.warning("wandb not available, skipping")
                self.use_wandb = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict with training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            input_lengths = batch['input_lengths']
            target_lengths = batch['target_lengths']

            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    if isinstance(self.model, CharLSTMSeq2Seq):
                        outputs = self.model(input_ids, target_ids, input_lengths, target_lengths)
                    else:  # Transformer
                        src_padding_mask = (input_ids == self.vocab.pad_token_id)
                        tgt_padding_mask = (target_ids == self.vocab.pad_token_id)
                        outputs = self.model(input_ids, target_ids, src_padding_mask, tgt_padding_mask)

                    # Loss (shift targets by 1 for next-token prediction)
                    loss = self.criterion(
                        outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1)),
                        target_ids[:, 1:].contiguous().view(-1)
                    )

                # Backward with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard training
                if isinstance(self.model, CharLSTMSeq2Seq):
                    outputs = self.model(input_ids, target_ids, input_lengths, target_lengths)
                else:  # Transformer
                    src_padding_mask = (input_ids == self.vocab.pad_token_id)
                    tgt_padding_mask = (target_ids == self.vocab.pad_token_id)
                    outputs = self.model(input_ids, target_ids, src_padding_mask, tgt_padding_mask)

                # Loss
                loss = self.criterion(
                    outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1)),
                    target_ids[:, 1:].contiguous().view(-1)
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()

            # Tracking
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            # Log to WandB
            if self.use_wandb and self.global_step % 10 == 0:
                self.wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })

        avg_loss = total_loss / num_batches

        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            input_lengths = batch['input_lengths']
            target_lengths = batch['target_lengths']

            # Forward pass (no teacher forcing for validation)
            if isinstance(self.model, CharLSTMSeq2Seq):
                outputs = self.model(input_ids, target_ids, input_lengths, target_lengths, teacher_forcing_ratio=0.0)
            else:  # Transformer
                src_padding_mask = (input_ids == self.vocab.pad_token_id)
                tgt_padding_mask = (target_ids == self.vocab.pad_token_id)
                outputs = self.model(input_ids, target_ids, src_padding_mask, tgt_padding_mask)

            # Loss
            loss = self.criterion(
                outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1)),
                target_ids[:, 1:].contiguous().view(-1)
            )

            total_loss += loss.item()
            num_batches += 1

            # Decode predictions for metrics
            predictions = outputs.argmax(dim=-1)  # [batch, seq_len]

            # Convert to text
            pred_texts = []
            ref_texts = []

            for i in range(predictions.size(0)):
                pred_ids = predictions[i, :target_lengths[i]].tolist()
                ref_ids = target_ids[i, 1:target_lengths[i]].tolist()

                pred_text = self.vocab.decode(pred_ids, skip_special_tokens=True)
                ref_text = self.vocab.decode(ref_ids, skip_special_tokens=True)

                pred_texts.append(pred_text)
                ref_texts.append(ref_text)

            # Update metrics
            self.val_metrics.update(pred_texts, ref_texts)

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()

        metrics['loss'] = avg_loss

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'vocab_size': self.vocab.vocab_size
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pt')
            self.logger.info(f"✅ Saved best model (CER: {metrics['cer']:.4f})")

        # Save epoch checkpoint
        if (self.current_epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pt')

    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {type(self.model).__name__}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['cer'])
                else:
                    self.scheduler.step()

            # Log
            epoch_time = time.time() - epoch_start_time

            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val CER: {val_metrics['cer']:.4f} | "
                f"Val Char Acc: {val_metrics['character_accuracy']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # WandB logging
            if self.use_wandb:
                self.wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss'],
                    'val/cer': val_metrics['cer'],
                    'val/char_accuracy': val_metrics['character_accuracy'],
                    'val/word_accuracy': val_metrics['word_accuracy'],
                    'val/cluster_validity': val_metrics['cluster_validity'],
                    'val/sequence_accuracy': val_metrics['sequence_accuracy'],
                })

            # Check if best model
            is_best = val_metrics['cer'] < self.best_val_cer

            if is_best:
                self.best_val_cer = val_metrics['cer']
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best=is_best)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered (patience={early_stopping_patience})")
                break

        self.logger.info(f"Training complete! Best validation CER: {self.best_val_cer:.4f}")

        if self.use_wandb:
            self.wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Khmer text correction model")

    # Data
    parser.add_argument('--data-dir', type=str, required=True, help='Training data directory')

    # Model
    parser.add_argument('--model', type=str, default='transformer', choices=['lstm', 'transformer'], help='Model type')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads (transformer only)')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--dim-feedforward', type=int, default=1024, help='FFN dimension (transformer only)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--mixed-precision', action='store_true', help='Use FP16 training')

    # Output
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for checkpoints')

    # WandB
    parser.add_argument('--wandb-project', type=str, default=None, help='WandB project name')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

    args = parser.parse_args()

    # Create vocab
    vocab = KhmerCharVocab()
    print(f"Vocabulary size: {vocab.vocab_size}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        Path(args.data_dir),
        vocab,
        batch_size=args.batch_size
    )

    # Create model
    if args.model == 'lstm':
        model = CharLSTMSeq2Seq(
            vocab_size=vocab.vocab_size,
            embed_dim=128,
            encoder_hidden_dim=args.d_model,
            decoder_hidden_dim=args.d_model,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_token_id=vocab.pad_token_id
        )
    else:  # transformer
        model = KhmerTransformerCorrector(
            vocab_size=vocab.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_seq_len=4000,
            pad_token_id=vocab.pad_token_id
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler (ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device(args.device),
        output_dir=Path(args.output_dir),
        use_wandb=args.wandb_project is not None,
        wandb_project=args.wandb_project,
        gradient_clip=args.gradient_clip,
        mixed_precision=args.mixed_precision
    )

    # Train
    trainer.train(args.epochs, args.early_stopping)


if __name__ == "__main__":
    main()
