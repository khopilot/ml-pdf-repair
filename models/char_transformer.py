"""
Character-Level Transformer for Khmer Text Correction

Production-grade encoder-decoder transformer for context-aware text reconstruction.
Based on "Attention is All You Need" (Vaswani et al., 2017) and
BART denoising architecture (Lewis et al., 2019).

Key Features:
    - Character-level processing
    - Multi-head self-attention
    - Cross-attention between encoder and decoder
    - Positional encoding for sequence order
    - Optimized for many-to-many character corruption patterns
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds position information to character embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: [batch, seq_len, d_model] - Input embeddings

        Returns:
            [batch, seq_len, d_model] - Embeddings with position info
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block.

    Self-attention + Feed-forward with residual connections and layer norm.
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float = 0.1):
        """
        Initialize encoder block.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: [batch, seq_len, d_model] - Input sequence
            src_mask: [seq_len, seq_len] - Attention mask
            src_key_padding_mask: [batch, seq_len] - Padding mask (True = ignore)

        Returns:
            [batch, seq_len, d_model] - Encoded sequence
        """
        # Self-attention with residual
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = self.norm1(src + self.dropout(attn_output))

        # Feedforward with residual
        ff_output = self.feedforward(src)
        src = self.norm2(src + ff_output)

        return src


class TransformerDecoderBlock(nn.Module):
    """
    Single transformer decoder block.

    Self-attention + Cross-attention + Feed-forward with residuals and layer norm.
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float = 0.1):
        """
        Initialize decoder block.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention (decoder attending to encoder)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tgt: [batch, tgt_len, d_model] - Target sequence
            memory: [batch, src_len, d_model] - Encoder output
            tgt_mask: [tgt_len, tgt_len] - Target attention mask (causal)
            memory_mask: [tgt_len, src_len] - Cross-attention mask
            tgt_key_padding_mask: [batch, tgt_len] - Target padding mask
            memory_key_padding_mask: [batch, src_len] - Source padding mask

        Returns:
            [batch, tgt_len, d_model] - Decoded sequence
        """
        # Self-attention with residual
        self_attn_output, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = self.norm1(tgt + self.dropout(self_attn_output))

        # Cross-attention with residual
        cross_attn_output, _ = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))

        # Feedforward with residual
        ff_output = self.feedforward(tgt)
        tgt = self.norm3(tgt + ff_output)

        return tgt


class KhmerTransformerCorrector(nn.Module):
    """
    Complete transformer model for Khmer PDF corruption correction.

    Character-level encoder-decoder architecture optimized for learning
    context-dependent character substitution patterns.
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 1024,
                 pad_token_id: int = 0):
        """
        Initialize transformer model.

        Args:
            vocab_size: Character vocabulary size (~150 for Khmer)
            d_model: Embedding/model dimension
            nhead: Number of attention heads
            num_encoder_layers: Encoder depth
            num_decoder_layers: Decoder depth
            dim_feedforward: FFN intermediate dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            pad_token_id: Padding token ID
        """
        super().__init__()

        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Character embeddings (shared between encoder and decoder)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self,
               src: torch.Tensor,
               src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: [batch, src_len] - Source token IDs
            src_key_padding_mask: [batch, src_len] - Padding mask (True = ignore)

        Returns:
            [batch, src_len, d_model] - Encoder output
        """
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Encode through all layers
        for layer in self.encoder_layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask)

        return src

    def decode(self,
               tgt: torch.Tensor,
               memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               tgt_key_padding_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode target sequence.

        Args:
            tgt: [batch, tgt_len] - Target token IDs
            memory: [batch, src_len, d_model] - Encoder output
            tgt_mask: [tgt_len, tgt_len] - Causal mask
            tgt_key_padding_mask: [batch, tgt_len] - Target padding mask
            memory_key_padding_mask: [batch, src_len] - Source padding mask

        Returns:
            [batch, tgt_len, d_model] - Decoder output
        """
        # Embed and add positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # Decode through all layers
        for layer in self.decoder_layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return tgt

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src: [batch, src_len] - Source (corrupted) token IDs
            tgt: [batch, tgt_len] - Target (correct) token IDs
            src_key_padding_mask: [batch, src_len] - Source padding (True = ignore)
            tgt_key_padding_mask: [batch, tgt_len] - Target padding (True = ignore)

        Returns:
            [batch, tgt_len, vocab_size] - Character logits
        """
        # Create causal mask for decoder (prevents attending to future tokens)
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, tgt.device)

        # Encode source
        memory = self.encode(src, src_key_padding_mask)

        # Decode target
        output = self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # Project to vocabulary
        logits = self.fc_out(output)

        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for decoder.

        Prevents position i from attending to positions j > i.

        Args:
            sz: Sequence length
            device: Torch device

        Returns:
            [sz, sz] mask (True = masked, False = allowed)
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask


if __name__ == "__main__":
    # Test model
    vocab_size = 200
    batch_size = 4
    src_len = 50
    tgt_len = 55

    model = KhmerTransformerCorrector(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )

    # Dummy data
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))

    # Create padding masks
    src_key_padding_mask = (src == 0)  # True = padding
    tgt_key_padding_mask = (tgt == 0)

    # Forward pass
    logits = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output shape: {logits.shape}")  # [batch, tgt_len, vocab_size]
    print(f"Memory (FP32): {sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.2f} MB")
