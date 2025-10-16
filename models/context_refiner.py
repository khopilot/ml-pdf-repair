"""
Context Refiner - Encoder-Decoder Transformer for structural corrections.

FIXED VERSION - Now properly handles variable-length input/target sequences
using encoder-decoder architecture with cross-attention (like the baseline).

Handles the 31.89% of errors that require context:
- Character insertions/deletions
- Cluster reordering
- Segmentation fixes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ContextRefiner(nn.Module):
    """
    Encoder-Decoder Transformer for contextual refinement.

    Takes atomic-mapped representations (from stage 1) and generates clean text
    via encoder-decoder architecture with cross-attention.

    KEY FIX: Output length = target_length (NOT source_length!)
    This allows handling variable-length input/target pairs.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = 4000,
                 pad_token_id: int = 0):
        """
        Initialize context refiner.

        Args:
            vocab_size: Character vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Encoder depth
            num_decoder_layers: Decoder depth
            dim_feedforward: FFN dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            pad_token_id: Padding token ID
        """
        super().__init__()

        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # ───────────────────────────────────────────────────────────
        # Target Embedding (for decoder input)
        # ───────────────────────────────────────────────────────────
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # ───────────────────────────────────────────────────────────
        # Positional Encoding
        # ───────────────────────────────────────────────────────────
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # ───────────────────────────────────────────────────────────
        # ENCODER (refine atomic-mapped source)
        # ───────────────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # ───────────────────────────────────────────────────────────
        # DECODER (generate clean text via cross-attention)
        # ───────────────────────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ───────────────────────────────────────────────────────────
        # Output Projection
        # ───────────────────────────────────────────────────────────
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: [batch, src_len, d_model] - Atomic-mapped representations
            tgt: [batch, tgt_len] - Target token IDs (for teacher forcing)
            src_key_padding_mask: [batch, src_len] - Source padding (True = ignore)
            tgt_key_padding_mask: [batch, tgt_len] - Target padding (True = ignore)

        Returns:
            [batch, tgt_len, vocab_size] - Character logits ✅ ALIGNED WITH TARGET!
        """
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Encode atomic-mapped source
        # ═══════════════════════════════════════════════════════════
        # src already has positional info from atomic mapper
        memory = self.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask
        )
        # memory: [batch, src_len, d_model]

        # ═══════════════════════════════════════════════════════════
        # STEP 2: Embed target sequence
        # ═══════════════════════════════════════════════════════════
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        # tgt_embedded: [batch, tgt_len, d_model]

        # ═══════════════════════════════════════════════════════════
        # STEP 3: Create causal mask for decoder
        # ═══════════════════════════════════════════════════════════
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, tgt.device)
        # tgt_mask: [tgt_len, tgt_len]

        # ═══════════════════════════════════════════════════════════
        # STEP 4: Decode (cross-attend to encoder memory)
        # ═══════════════════════════════════════════════════════════
        output = self.decoder(
            tgt_embedded,                    # [batch, tgt_len, d_model]
            memory,                          # [batch, src_len, d_model]  ← Different length!
            tgt_mask=tgt_mask,               # [tgt_len, tgt_len]
            tgt_key_padding_mask=tgt_key_padding_mask,     # [batch, tgt_len]
            memory_key_padding_mask=src_key_padding_mask   # [batch, src_len]
        )
        # output: [batch, tgt_len, d_model]  ✅ Aligned with target!

        # ═══════════════════════════════════════════════════════════
        # STEP 5: Project to vocabulary
        # ═══════════════════════════════════════════════════════════
        logits = self.fc_out(output)
        # logits: [batch, tgt_len, vocab_size]  ✅ CORRECT SHAPE!

        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for decoder.

        Prevents position i from attending to positions j > i.

        Returns:
            [sz, sz] mask (True = masked, False = allowed)
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask


if __name__ == "__main__":
    # Test context refiner
    print("Testing ContextRefiner (Encoder-Decoder)...")

    vocab_size = 200
    batch_size = 4
    src_len = 150  # Source length (from atomic mapper)
    tgt_len = 95   # Target length (different from source!)

    refiner = ContextRefiner(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512
    )

    print(f"Parameters: {sum(p.numel() for p in refiner.parameters()):,}")

    # Test forward pass
    src = torch.randn(batch_size, src_len, 128)  # From atomic mapper
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))

    src_padding_mask = (src.sum(dim=-1) == 0)
    tgt_padding_mask = (tgt == 0)

    logits = refiner(src, tgt, src_padding_mask, tgt_padding_mask)

    print(f"\nInput shapes:")
    print(f"  src: {src.shape} (from atomic mapper)")
    print(f"  tgt: {tgt.shape} (target sequence)")
    print(f"\nOutput shape: {logits.shape}")
    print(f"Expected: [{batch_size}, {tgt_len}, {vocab_size}]")
    print(f"Match: {logits.shape == torch.Size([batch_size, tgt_len, vocab_size])}")

    print("\n✅ ContextRefiner now produces [batch, tgt_len, vocab_size]!")
    print("   (Previously produced [batch, src_len, vocab_size] - WRONG)")
