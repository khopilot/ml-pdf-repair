"""
Character-Level LSTM Seq2Seq for Khmer Text Correction

Baseline model implementing encoder-decoder architecture with attention.
Serves as proof-of-concept before scaling to Transformer.

Architecture:
    - Bidirectional LSTM encoder
    - Bahdanau attention mechanism
    - LSTM decoder with attention
    - Character-level processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Encoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    Encodes input sequence (corrupted text) into hidden representations.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Initialize encoder.

        Args:
            vocab_size: Character vocabulary size
            embed_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension (will be doubled due to bidirectional)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Character embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                input_ids: torch.Tensor,
                input_lengths: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input sequence.

        Args:
            input_ids: [batch, seq_len] - Input token IDs
            input_lengths: [batch] - Actual lengths (before padding)

        Returns:
            encoder_outputs: [batch, seq_len, hidden_dim * 2] - Hidden states
            hidden_states: (h_n, c_n) - Final LSTM states
        """
        # Embed
        embedded = self.dropout(self.embedding(input_ids))  # [batch, seq_len, embed_dim]

        # Pack sequence (ignore padding)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # LSTM forward pass
        packed_outputs, hidden_states = self.lstm(packed)

        # Unpack
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )

        return encoder_outputs, hidden_states


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    Computes attention weights over encoder outputs.
    """

    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int):
        """
        Initialize attention.

        Args:
            encoder_hidden_dim: Encoder hidden dimension (bidirectional, so 2x)
            decoder_hidden_dim: Decoder hidden dimension
        """
        super().__init__()

        self.W_encoder = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        self.W_decoder = nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
        self.V = nn.Linear(decoder_hidden_dim, 1)

    def forward(self,
                decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden: [batch, decoder_hidden_dim] - Current decoder state
            encoder_outputs: [batch, src_len, encoder_hidden_dim] - Encoder outputs
            mask: [batch, src_len] - Mask for padding (1 = valid, 0 = padding)

        Returns:
            context: [batch, encoder_hidden_dim] - Context vector
            attention_weights: [batch, src_len] - Attention weights
        """
        batch_size, src_len, _ = encoder_outputs.size()

        # Compute attention scores
        # decoder_hidden: [batch, decoder_hidden] → [batch, 1, decoder_hidden]
        decoder_hidden = decoder_hidden.unsqueeze(1)  # [batch, 1, decoder_hidden]

        # encoder_outputs: [batch, src_len, encoder_hidden]
        # W_encoder(encoder_outputs): [batch, src_len, decoder_hidden]
        encoder_transformed = self.W_encoder(encoder_outputs)

        # decoder_hidden: [batch, 1, decoder_hidden]
        # W_decoder(decoder_hidden): [batch, 1, decoder_hidden]
        decoder_transformed = self.W_decoder(decoder_hidden)

        # Additive attention: tanh(W_encoder * encoder + W_decoder * decoder)
        energy = torch.tanh(encoder_transformed + decoder_transformed)  # [batch, src_len, decoder_hidden]

        # Score: V * energy
        attention_scores = self.V(energy).squeeze(2)  # [batch, src_len]

        # Mask padding positions
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, src_len]

        # Context vector: weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch, 1, src_len]
            encoder_outputs                   # [batch, src_len, encoder_hidden]
        ).squeeze(1)  # [batch, encoder_hidden]

        return context, attention_weights


class Decoder(nn.Module):
    """
    LSTM decoder with attention.

    Generates corrected text character-by-character with attention over encoder outputs.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 encoder_hidden_dim: int = 512,  # Bidirectional encoder: 256*2
                 num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Initialize decoder.

        Args:
            vocab_size: Character vocabulary size
            embed_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            encoder_hidden_dim: Encoder output dimension (2x if bidirectional)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Character embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Attention
        self.attention = BahdanauAttention(encoder_hidden_dim, hidden_dim)

        # LSTM (input = embedding + context)
        self.lstm = nn.LSTM(
            embed_dim + encoder_hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim + encoder_hidden_dim + embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward_step(self,
                     input_token: torch.Tensor,
                     hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoder_outputs: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Single decoder step.

        Args:
            input_token: [batch, 1] - Current input token
            hidden: (h, c) - LSTM hidden state
            encoder_outputs: [batch, src_len, encoder_hidden] - Encoder outputs
            mask: [batch, src_len] - Padding mask

        Returns:
            output: [batch, vocab_size] - Character logits
            hidden: (h, c) - Updated LSTM state
            attention_weights: [batch, src_len] - Attention weights
        """
        # Embed input token
        embedded = self.dropout(self.embedding(input_token))  # [batch, 1, embed_dim]

        # Compute attention
        # Use top-layer hidden state for attention query
        decoder_hidden = hidden[0][-1]  # [batch, hidden_dim]
        context, attention_weights = self.attention(decoder_hidden, encoder_outputs, mask)

        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # [batch, 1, embed_dim + encoder_hidden]

        # LSTM step
        lstm_output, hidden = self.lstm(lstm_input, hidden)  # lstm_output: [batch, 1, hidden_dim]

        # Concatenate lstm_output, context, embedding for output
        output_input = torch.cat([
            lstm_output.squeeze(1),  # [batch, hidden_dim]
            context,                 # [batch, encoder_hidden]
            embedded.squeeze(1)      # [batch, embed_dim]
        ], dim=1)  # [batch, hidden_dim + encoder_hidden + embed_dim]

        # Project to vocabulary
        output = self.fc_out(output_input)  # [batch, vocab_size]

        return output, hidden, attention_weights


class CharLSTMSeq2Seq(nn.Module):
    """
    Complete LSTM Seq2Seq model with attention.

    Baseline model for Khmer PDF corruption correction.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 encoder_hidden_dim: int = 256,
                 decoder_hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 pad_token_id: int = 0):
        """
        Initialize seq2seq model.

        Args:
            vocab_size: Character vocabulary size
            embed_dim: Embedding dimension
            encoder_hidden_dim: Encoder LSTM hidden dimension (will be 2x due to bidirectional)
            decoder_hidden_dim: Decoder LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pad_token_id: Padding token ID
        """
        super().__init__()

        self.pad_token_id = pad_token_id

        # Encoder
        self.encoder = Encoder(
            vocab_size,
            embed_dim,
            encoder_hidden_dim,
            num_layers,
            dropout
        )

        # Decoder
        self.decoder = Decoder(
            vocab_size,
            embed_dim,
            decoder_hidden_dim,
            encoder_hidden_dim * 2,  # Bidirectional encoder
            num_layers,
            dropout
        )

        # Bridge encoder hidden to decoder hidden
        self.bridge = nn.Linear(encoder_hidden_dim * 2 * num_layers, decoder_hidden_dim * num_layers)

    def forward(self,
                input_ids: torch.Tensor,
                target_ids: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            input_ids: [batch, src_len] - Input (corrupted) text
            target_ids: [batch, tgt_len] - Target (correct) text
            input_lengths: [batch] - Input lengths
            target_lengths: [batch] - Target lengths
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: [batch, tgt_len, vocab_size] - Character logits
        """
        batch_size = input_ids.size(0)
        tgt_len = target_ids.size(1)
        vocab_size = self.decoder.fc_out.out_features

        # Encode
        encoder_outputs, encoder_hidden = self.encoder(input_ids, input_lengths)

        # Create padding mask for attention
        mask = (input_ids != self.pad_token_id).float()  # [batch, src_len]

        # Initialize decoder hidden state from encoder
        # Concatenate forward and backward final hidden states
        h_n, c_n = encoder_hidden
        # h_n: [num_layers*2, batch, hidden_dim] → [batch, num_layers*2*hidden_dim]
        h_n = h_n.transpose(0, 1).contiguous().view(batch_size, -1)
        c_n = c_n.transpose(0, 1).contiguous().view(batch_size, -1)

        # Bridge to decoder hidden dimension
        decoder_h = self.bridge(h_n).view(batch_size, self.decoder.num_layers, -1).transpose(0, 1).contiguous()
        decoder_c = self.bridge(c_n).view(batch_size, self.decoder.num_layers, -1).transpose(0, 1).contiguous()
        decoder_hidden = (decoder_h, decoder_c)

        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=input_ids.device)

        # First input to decoder is <START> token (assumed to be target_ids[:, 0])
        input_token = target_ids[:, 0].unsqueeze(1)  # [batch, 1]

        for t in range(1, tgt_len):
            # Decoder step
            output, decoder_hidden, attention_weights = self.decoder.forward_step(
                input_token,
                decoder_hidden,
                encoder_outputs,
                mask
            )

            # Store output
            outputs[:, t] = output

            # Teacher forcing: use ground truth or predicted token
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            if teacher_force:
                input_token = target_ids[:, t].unsqueeze(1)  # [batch, 1]
            else:
                input_token = output.argmax(dim=1).unsqueeze(1)  # [batch, 1]

        return outputs


if __name__ == "__main__":
    # Test model
    vocab_size = 200
    batch_size = 4
    src_len = 50
    tgt_len = 55

    model = CharLSTMSeq2Seq(
        vocab_size=vocab_size,
        embed_dim=128,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )

    # Dummy data
    input_ids = torch.randint(1, vocab_size, (batch_size, src_len))
    target_ids = torch.randint(1, vocab_size, (batch_size, tgt_len))
    input_lengths = torch.tensor([src_len] * batch_size)
    target_lengths = torch.tensor([tgt_len] * batch_size)

    # Forward pass
    outputs = model(input_ids, target_ids, input_lengths, target_lengths)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output shape: {outputs.shape}")  # [batch, tgt_len, vocab_size]
