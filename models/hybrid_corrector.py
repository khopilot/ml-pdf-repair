"""
Hybrid Corrector - Atomic Mapper + Context Refiner.

Two-stage architecture for Khmer PDF text correction:
1. Atomic Mapper: Fixes 68% of errors (character substitutions)
2. Context Refiner: Fixes 32% of errors (insertions, deletions, reordering)

This compositional approach is more sample-efficient and interpretable
than end-to-end seq2seq models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .atomic_mapper import AtomicCharMapper
from .context_refiner import ContextRefiner


class HybridCorrector(nn.Module):
    """
    Hybrid two-stage correction model (SIMPLIFIED & FIXED).

    Stage 1: Atomic mapper - Maps corrupt characters to d_model representations
    Stage 2: Context refiner - Encoder-decoder that generates clean text

    NOW WORKS with variable-length input/target pairs!
    """

    def __init__(self,
                 vocab_size: int,
                 # Atomic mapper params
                 atomic_embed_dim: int = 128,
                 atomic_hidden_dim: int = 256,
                 atomic_num_layers: int = 3,
                 # Context refiner params
                 refiner_d_model: int = 128,
                 refiner_nhead: int = 4,
                 refiner_num_encoder_layers: int = 3,
                 refiner_num_decoder_layers: int = 3,
                 refiner_dim_feedforward: int = 512,
                 # Shared params
                 dropout: float = 0.1,
                 max_seq_len: int = 4000,
                 pad_token_id: int = 0):
        """
        Initialize hybrid corrector.

        Args:
            vocab_size: Character vocabulary size
            atomic_embed_dim: Atomic mapper embedding dimension
            atomic_hidden_dim: Atomic mapper hidden dimension
            atomic_num_layers: Atomic mapper MLP layers
            refiner_d_model: Context refiner model dimension
            refiner_nhead: Context refiner attention heads
            refiner_num_encoder_layers: Refiner encoder layers
            refiner_num_decoder_layers: Refiner decoder layers
            refiner_dim_feedforward: Context refiner FFN dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            pad_token_id: Padding token ID
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.d_model = refiner_d_model

        # ───────────────────────────────────────────────────────────
        # Stage 1: Atomic character mapper
        # ───────────────────────────────────────────────────────────
        # Maps corrupted text to d_model dimensional representations
        self.atomic_mapper = AtomicCharMapper(
            vocab_size=vocab_size,
            embed_dim=refiner_d_model,  # Match refiner dimension!
            hidden_dim=atomic_hidden_dim,
            num_layers=atomic_num_layers,
            dropout=dropout,
            pad_token_id=pad_token_id
        )

        # Projection from atomic mapper output to refiner input
        # (in case dimensions don't match)
        self.mapper_projection = nn.Linear(vocab_size, refiner_d_model)

        # ───────────────────────────────────────────────────────────
        # Stage 2: Context refiner (encoder-decoder)
        # ───────────────────────────────────────────────────────────
        self.context_refiner = ContextRefiner(
            vocab_size=vocab_size,
            d_model=refiner_d_model,
            nhead=refiner_nhead,
            num_encoder_layers=refiner_num_encoder_layers,
            num_decoder_layers=refiner_num_decoder_layers,
            dim_feedforward=refiner_dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id
        )

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass (SIMPLIFIED - like baseline!).

        Args:
            src: [batch, src_len] - Source (corrupted) token IDs
            tgt: [batch, tgt_len] - Target (clean) token IDs
            src_key_padding_mask: [batch, src_len] - Source padding
            tgt_key_padding_mask: [batch, tgt_len] - Target padding

        Returns:
            [batch, tgt_len, vocab_size] - Character logits ✅ ALIGNED WITH TARGET!
        """
        # ═══════════════════════════════════════════════════════════
        # STAGE 1: Atomic character mapping
        # ═══════════════════════════════════════════════════════════
        atomic_logits, _ = self.atomic_mapper(src, return_confidence=False)
        # atomic_logits: [batch, src_len, vocab_size]

        # Project to refiner dimension
        mapped = self.mapper_projection(atomic_logits)
        # mapped: [batch, src_len, d_model]

        # ═══════════════════════════════════════════════════════════
        # STAGE 2: Context refinement (encoder-decoder)
        # ═══════════════════════════════════════════════════════════
        logits = self.context_refiner(
            src=mapped,                      # [batch, src_len, d_model]
            tgt=tgt,                         # [batch, tgt_len]
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        # logits: [batch, tgt_len, vocab_size]  ✅ CORRECT!

        return logits

    def get_stage_parameters(self) -> Tuple[list, list]:
        """
        Get parameters for each stage separately (for staged training).

        Returns:
            atomic_params: Parameters of atomic mapper
            refiner_params: Parameters of context refiner
        """
        atomic_params = list(self.atomic_mapper.parameters()) + list(self.mapper_projection.parameters())
        refiner_params = list(self.context_refiner.parameters())

        return atomic_params, refiner_params


class EnhancedHybridCorrector(nn.Module):
    """
    Enhanced hybrid corrector with context-aware atomic mapper.

    Uses ContextAwareAtomicMapper (with local context window) instead
    of vanilla AtomicCharMapper.
    """

    def __init__(self,
                 vocab_size: int,
                 # Atomic mapper params
                 atomic_embed_dim: int = 128,
                 atomic_hidden_dim: int = 256,
                 atomic_num_layers: int = 3,
                 atomic_context_window: int = 3,
                 # Context refiner params (seq2seq)
                 refiner_d_model: int = 128,
                 refiner_nhead: int = 4,
                 refiner_num_encoder_layers: int = 3,
                 refiner_num_decoder_layers: int = 3,
                 refiner_dim_feedforward: int = 512,
                 # Shared
                 dropout: float = 0.1,
                 max_seq_len: int = 4000,
                 pad_token_id: int = 0):
        """Initialize enhanced hybrid corrector."""
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # Context-aware atomic mapper
        self.atomic_mapper = ContextAwareAtomicMapper(
            vocab_size=vocab_size,
            embed_dim=atomic_embed_dim,
            hidden_dim=atomic_hidden_dim,
            num_layers=atomic_num_layers,
            context_window=atomic_context_window,
            dropout=dropout,
            pad_token_id=pad_token_id
        )

        # Seq2seq context refiner
        self.context_refiner = SequenceToSequenceRefiner(
            vocab_size=vocab_size,
            d_model=refiner_d_model,
            nhead=refiner_nhead,
            num_encoder_layers=refiner_num_encoder_layers,
            num_decoder_layers=refiner_num_decoder_layers,
            dim_feedforward=refiner_dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id
        )

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src: [batch, src_len] - Corrupted input
            tgt: [batch, tgt_len] - Target output
            src_key_padding_mask: [batch, src_len] - Source padding
            tgt_key_padding_mask: [batch, tgt_len] - Target padding

        Returns:
            [batch, tgt_len, vocab_size] - Output logits
        """
        # Stage 1: Atomic pre-correction
        atomic_logits, _ = self.atomic_mapper(src, return_confidence=False)
        atomic_preds = atomic_logits.argmax(dim=-1)

        # Stage 2: Seq2seq refinement
        # Use atomic predictions as source for refiner
        refined_logits = self.context_refiner(
            atomic_preds,
            tgt,
            src_key_padding_mask,
            tgt_key_padding_mask
        )

        return refined_logits


if __name__ == "__main__":
    print("Testing HybridCorrector...")

    vocab_size = 200
    batch_size = 4
    seq_len = 50

    # Create hybrid model
    model = HybridCorrector(
        vocab_size=vocab_size,
        atomic_embed_dim=128,
        atomic_hidden_dim=256,
        atomic_num_layers=3,
        refiner_d_model=128,
        refiner_nhead=4,
        refiner_num_layers=3,
        refiner_dim_feedforward=512,
        use_gating=True
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    atomic_params, refiner_params = model.get_stage_parameters()
    print(f"Atomic mapper parameters: {sum(p.numel() for p in atomic_params):,}")
    print(f"Context refiner parameters: {sum(p.numel() for p in refiner_params):,}")

    # Test forward pass
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    padding_mask = (input_ids == 0)

    # Training mode
    model.train()
    logits, intermediate = model(input_ids, padding_mask, return_intermediate=True)

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Atomic confidence (mean): {intermediate['atomic_confidence'].mean():.4f}")
    print(f"Needs refinement: {intermediate['needs_refinement'].float().mean():.2%}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        logits, _ = model(input_ids, padding_mask)
    print(f"\nInference output shape: {logits.shape}")

    # Test enhanced hybrid
    print("\n" + "="*60)
    print("Testing EnhancedHybridCorrector...")

    enhanced_model = EnhancedHybridCorrector(
        vocab_size=vocab_size,
        atomic_embed_dim=128,
        atomic_hidden_dim=256,
        atomic_num_layers=3,
        atomic_context_window=3,
        refiner_d_model=128,
        refiner_nhead=4,
        refiner_num_encoder_layers=3,
        refiner_num_decoder_layers=3,
        refiner_dim_feedforward=512
    )

    print(f"Total parameters: {sum(p.numel() for p in enhanced_model.parameters()):,}")

    tgt_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    logits = enhanced_model(input_ids, tgt_ids)

    print(f"Output shape: {logits.shape}")
    print(f"\nMemory comparison:")
    print(f"  Hybrid: {sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.2f} MB")
    print(f"  Enhanced: {sum(p.numel() * 4 for p in enhanced_model.parameters()) / 1024**2:.2f} MB")
