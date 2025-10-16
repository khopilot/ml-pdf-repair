"""
Atomic Character Mapper for Khmer text correction.

Learns character-level substitution patterns with confidence scores.
Handles the 48.71% of errors that are simple substitutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AtomicCharMapper(nn.Module):
    """
    Character-level substitution network.

    Learns probabilistic mappings for atomic character corrections.
    Unlike a lookup table, this can handle ambiguous cases where
    one corrupt character maps to multiple possible corrections.

    Architecture:
        char_id -> embedding -> MLP -> vocab_logits

    Key properties:
        - Position-independent (same char, same correction)
        - Fast (no attention, pure feed-forward)
        - Interpretable (can extract substitution table)
        - Confidence-aware (outputs probabilities)
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 pad_token_id: int = 0):
        """
        Initialize atomic mapper.

        Args:
            vocab_size: Character vocabulary size
            embed_dim: Character embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
            pad_token_id: Padding token ID
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        # MLP for character substitution
        layers = []
        input_dim = embed_dim

        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else vocab_size
            layers.append(nn.Linear(input_dim, output_dim))

            if i < num_layers - 1:
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            input_dim = output_dim

        self.mapper = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Xavier uniform for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                input_ids: torch.Tensor,
                return_confidence: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for atomic character mapping.

        Args:
            input_ids: [batch, seq_len] - Character token IDs
            return_confidence: Return confidence scores

        Returns:
            logits: [batch, seq_len, vocab_size] - Character probabilities
            confidence: [batch, seq_len] - Confidence scores (optional)
        """
        # Embed characters
        # [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(input_ids)

        # Map to substitutions (position-independent)
        # [batch, seq_len, embed_dim] -> [batch, seq_len, vocab_size]
        logits = self.mapper(embedded)

        if return_confidence:
            # Confidence = max probability after softmax
            probs = F.softmax(logits, dim=-1)
            confidence, _ = probs.max(dim=-1)
            return logits, confidence
        else:
            return logits, None

    def get_substitution_table(self, temperature: float = 1.0) -> torch.Tensor:
        """
        Extract learned substitution table.

        Args:
            temperature: Softmax temperature (lower = sharper)

        Returns:
            [vocab_size, vocab_size] substitution probability matrix
            where table[i, j] = P(correct_char=j | corrupt_char=i)
        """
        # Get embeddings for all vocab
        all_ids = torch.arange(self.vocab_size, device=next(self.parameters()).device)
        embedded = self.embedding(all_ids)  # [vocab_size, embed_dim]

        # Get logits
        logits = self.mapper(embedded)  # [vocab_size, vocab_size]

        # Apply temperature and softmax
        probs = F.softmax(logits / temperature, dim=-1)

        return probs

    def predict_greedy(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Greedy decoding (argmax).

        Args:
            input_ids: [batch, seq_len] - Input character IDs

        Returns:
            [batch, seq_len] - Corrected character IDs
        """
        logits, _ = self.forward(input_ids, return_confidence=False)
        return logits.argmax(dim=-1)

    def predict_with_confidence(self,
                                 input_ids: torch.Tensor,
                                 confidence_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with confidence scores.

        Args:
            input_ids: [batch, seq_len] - Input character IDs
            confidence_threshold: Threshold for flagging low-confidence

        Returns:
            predictions: [batch, seq_len] - Corrected character IDs
            needs_refinement: [batch, seq_len] - Boolean mask of low-confidence chars
        """
        logits, confidence = self.forward(input_ids, return_confidence=True)
        predictions = logits.argmax(dim=-1)
        needs_refinement = confidence < confidence_threshold

        return predictions, needs_refinement


class ContextAwareAtomicMapper(nn.Module):
    """
    Atomic mapper with local context window.

    Extension of AtomicCharMapper that considers neighboring characters.
    Useful for handling context-dependent substitutions (e.g., cluster reordering).
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 context_window: int = 3,
                 dropout: float = 0.1,
                 pad_token_id: int = 0):
        """
        Initialize context-aware mapper.

        Args:
            vocab_size: Character vocabulary size
            embed_dim: Character embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            context_window: Window size (e.g., 3 = look at ±1 chars)
            dropout: Dropout probability
            pad_token_id: Padding token ID
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.pad_token_id = pad_token_id

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        # 1D convolution for local context
        self.context_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=hidden_dim,
            kernel_size=context_window,
            padding=context_window // 2
        )

        # MLP for mapping
        layers = []
        input_dim = hidden_dim

        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else vocab_size
            layers.append(nn.Linear(input_dim, output_dim))

            if i < num_layers - 1:
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            input_dim = output_dim

        self.mapper = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                input_ids: torch.Tensor,
                return_confidence: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with local context.

        Args:
            input_ids: [batch, seq_len] - Input character IDs
            return_confidence: Return confidence scores

        Returns:
            logits: [batch, seq_len, vocab_size]
            confidence: [batch, seq_len] (optional)
        """
        # Embed
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        # Apply convolution for local context
        # Conv1d expects [batch, channels, seq_len]
        embedded_t = embedded.transpose(1, 2)  # [batch, embed_dim, seq_len]
        context = self.context_conv(embedded_t)  # [batch, hidden_dim, seq_len]
        context = F.relu(context)
        context = context.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # Map to substitutions
        logits = self.mapper(context)  # [batch, seq_len, vocab_size]

        if return_confidence:
            probs = F.softmax(logits, dim=-1)
            confidence, _ = probs.max(dim=-1)
            return logits, confidence
        else:
            return logits, None


if __name__ == "__main__":
    # Test atomic mapper
    print("Testing AtomicCharMapper...")

    vocab_size = 200
    batch_size = 4
    seq_len = 50

    # Create model
    mapper = AtomicCharMapper(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=3
    )

    print(f"Parameters: {sum(p.numel() for p in mapper.parameters()):,}")

    # Test forward pass
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    logits, confidence = mapper(input_ids, return_confidence=True)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Mean confidence: {confidence.mean():.4f}")

    # Test substitution table extraction
    sub_table = mapper.get_substitution_table()
    print(f"\nSubstitution table shape: {sub_table.shape}")
    print(f"Example: char 10 -> {sub_table[10].argmax().item()} (conf: {sub_table[10].max():.4f})")

    # Test context-aware mapper
    print("\n" + "="*60)
    print("Testing ContextAwareAtomicMapper...")

    context_mapper = ContextAwareAtomicMapper(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=3,
        context_window=3
    )

    print(f"Parameters: {sum(p.numel() for p in context_mapper.parameters()):,}")

    logits, confidence = context_mapper(input_ids, return_confidence=True)
    print(f"Output logits shape: {logits.shape}")
    print(f"Mean confidence: {confidence.mean():.4f}")
