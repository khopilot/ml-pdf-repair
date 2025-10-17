#!/usr/bin/env python3
"""
Training script for Hybrid Corrector model.

Supports:
- Two-stage training (atomic first, then refiner)
- End-to-end joint training
- Staged learning rates
- WandB logging
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
from models.atomic_mapper import AtomicCharMapper
from models.context_refiner import ContextRefiner
from models.hybrid_corrector import HybridCorrector
from training.metrics import MetricCollection


class HybridTrainer:
    """
    Trainer for hybrid atomic + context correction model.

    Supports multiple training strategies:
    1. Atomic-first: Pre-train atomic mapper, freeze, train refiner
    2. Joint: Train both stages simultaneously
    3. Fine-tune: Pre-train separately, then fine-tune end-to-end
    """

    def __init__(self,
                 model: HybridCorrector,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 vocab: KhmerCharVocab,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 device: torch.device,
                 output_dir: Path,
                 use_wandb: bool = False,
                 wandb_project: Optional[str] = None,
                 gradient_clip: float = 1.0):
        """Initialize trainer."""
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

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)

        # Tracking
        self.best_val_cer = float('inf')
        self.current_epoch = 0
        self.global_step = 0

        # Metrics
        self.train_metrics = MetricCollection()
        self.val_metrics = MetricCollection()

        # Mixed Precision Training (BF16) for A100 CUDA
        # BF16 has better stability than FP16 and doesn't need gradient scaling
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            logging.info("🚀 Mixed precision (BF16) training enabled for A100")
        else:
            logging.info("⚠️ Running in FP32 mode (no CUDA)")

        # WandB
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, config={
                    'model': 'HybridCorrector',
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
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            input_lengths = batch['input_lengths']
            target_lengths = batch['target_lengths']

            # Create padding masks
            src_padding_mask = (input_ids == self.vocab.pad_token_id)

            # CRITICAL FIX: Prepare decoder input (shifted right with BOS)
            # tgt_out = ground truth to predict: [y1, y2, y3, ..., yn]
            # tgt_in  = decoder input: [BOS, y1, y2, ..., yn-1]
            BOS = self.vocab.start_token_id
            batch_size, tgt_len = target_ids.size()
            bos_col = torch.full((batch_size, 1), BOS, dtype=target_ids.dtype, device=target_ids.device)
            tgt_in = torch.cat([bos_col, target_ids[:, :-1]], dim=1)  # Shift right
            tgt_out = target_ids  # Ground truth

            tgt_padding_mask = (tgt_in == self.vocab.pad_token_id)

            # Mixed precision training (BF16 for A100)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                # Forward pass with CORRECT target shift
                outputs = self.model(
                    input_ids,           # [batch, src_len]
                    tgt_in,              # [batch, tgt_len] - decoder input with BOS
                    src_padding_mask,    # [batch, src_len]
                    tgt_padding_mask     # [batch, tgt_len]
                )
                # outputs: [batch, tgt_len, vocab_size]

                # Loss: predict tgt_out from model outputs
                # No shifting needed - already aligned!
                loss = self.criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    tgt_out.contiguous().view(-1)
                )

            # Backward pass (BF16 doesn't need gradient scaling)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            # Tracking
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Aggressive GPU cache clearing (every 4 batches)
            if num_batches % 4 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})

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
        """Validate on validation set."""
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

            src_padding_mask = (input_ids == self.vocab.pad_token_id)

            # CRITICAL FIX: Match training - use shifted decoder input
            BOS = self.vocab.start_token_id
            batch_size, tgt_len = target_ids.size()
            bos_col = torch.full((batch_size, 1), BOS, dtype=target_ids.dtype, device=target_ids.device)
            tgt_in = torch.cat([bos_col, target_ids[:, :-1]], dim=1)
            tgt_out = target_ids

            tgt_padding_mask = (tgt_in == self.vocab.pad_token_id)

            # Mixed precision validation (BF16 for A100)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                # Forward pass matching training
                outputs = self.model(
                    input_ids,
                    tgt_in,           # Shifted input
                    src_padding_mask,
                    tgt_padding_mask
                )
                # outputs: [batch, tgt_len, vocab_size]

                # Loss: predict tgt_out
                loss = self.criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    tgt_out.contiguous().view(-1)
                )

            total_loss += loss.item()
            num_batches += 1

            # Decode predictions for metrics (from teacher-forced output)
            predictions = outputs.argmax(dim=-1)

            # Convert to text
            pred_texts = []
            ref_texts = []

            for i in range(predictions.size(0)):
                pred_ids = predictions[i, :target_lengths[i]].tolist()
                ref_ids = target_ids[i, :target_lengths[i]].tolist()

                pred_text = self.vocab.decode(pred_ids, skip_special_tokens=True)
                ref_text = self.vocab.decode(ref_ids, skip_special_tokens=True)

                pred_texts.append(pred_text)
                ref_texts.append(ref_text)

            # Update metrics
            self.val_metrics.update(pred_texts, ref_texts)

            progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})

        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()
        metrics['loss'] = avg_loss

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
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

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load checkpoint and return starting epoch."""
        if not checkpoint_path.exists():
            self.logger.info("No checkpoint found, starting from scratch")
            return 0

        try:
            self.logger.info(f"📂 Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Restore model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Restore training state
            start_epoch = checkpoint['epoch'] + 1  # Start from NEXT epoch
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_cer = checkpoint['metrics'].get('cer', float('inf'))

            self.logger.info(f"✅ Resumed from epoch {checkpoint['epoch']}")
            self.logger.info(f"   Best CER so far: {self.best_val_cer:.4f}")
            self.logger.info(f"   Global step: {self.global_step}")

            return start_epoch
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            self.logger.info("Starting from scratch")
            return 0

    def train(self, num_epochs: int, early_stopping_patience: int = 10, resume: bool = True):
        """Main training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: HybridCorrector")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        atomic_params, refiner_params = self.model.get_stage_parameters()
        self.logger.info(f"  Atomic mapper: {sum(p.numel() for p in atomic_params):,}")
        self.logger.info(f"  Context refiner: {sum(p.numel() for p in refiner_params):,}")

        # Resume from checkpoint if exists
        start_epoch = 0
        if resume:
            checkpoint_path = self.output_dir / 'latest_checkpoint.pt'
            start_epoch = self.load_checkpoint(checkpoint_path)
            if start_epoch > 0:
                self.logger.info(f"🔄 Resuming training from epoch {start_epoch}")

        patience_counter = 0

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            try:
                # Train
                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.error(f"💥 GPU OUT OF MEMORY at Epoch {epoch + 1}")
                    self.logger.error(f"Error: {e}")
                    self.logger.error(f"Try reducing BATCH_SIZE (current: {self.train_loader.batch_size})")
                    if torch.cuda.is_available():
                        self.logger.error(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                        self.logger.error(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                        torch.cuda.empty_cache()
                    raise
                else:
                    self.logger.error(f"💥 CUDA Error at Epoch {epoch + 1}: {e}")
                    raise

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

            # Clear GPU cache to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered (patience={early_stopping_patience})")
                break

        self.logger.info(f"Training complete! Best validation CER: {self.best_val_cer:.4f}")

        if self.use_wandb:
            self.wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Corrector")

    # Data
    parser.add_argument('--data-dir', type=str, required=True, help='Training data directory')

    # Model
    parser.add_argument('--atomic-embed-dim', type=int, default=128, help='Atomic mapper embedding dim')
    parser.add_argument('--atomic-hidden-dim', type=int, default=256, help='Atomic mapper hidden dim')
    parser.add_argument('--atomic-layers', type=int, default=3, help='Atomic mapper layers')
    parser.add_argument('--refiner-d-model', type=int, default=128, help='Refiner model dim')
    parser.add_argument('--refiner-nhead', type=int, default=4, help='Refiner attention heads')
    parser.add_argument('--refiner-layers', type=int, default=3, help='Refiner layers')
    parser.add_argument('--refiner-ffn-dim', type=int, default=512, help='Refiner FFN dim')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    # Output
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')

    # WandB
    parser.add_argument('--wandb-project', type=str, default=None, help='WandB project name')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

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
    model = HybridCorrector(
        vocab_size=vocab.vocab_size,
        atomic_embed_dim=args.atomic_embed_dim,
        atomic_hidden_dim=args.atomic_hidden_dim,
        atomic_num_layers=args.atomic_layers,
        refiner_d_model=args.refiner_d_model,
        refiner_nhead=args.refiner_nhead,
        refiner_num_encoder_layers=args.refiner_layers,
        refiner_num_decoder_layers=args.refiner_layers,
        refiner_dim_feedforward=args.refiner_ffn_dim,
        dropout=args.dropout,
        pad_token_id=vocab.pad_token_id
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Trainer
    trainer = HybridTrainer(
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
        gradient_clip=args.gradient_clip
    )

    # Train
    trainer.train(args.epochs, args.early_stopping, resume=args.resume)


if __name__ == "__main__":
    main()
