"""
Production inference API for Khmer PDF corruption correction.

Provides simple API for correcting corrupted text using trained models.

Usage:
    from inference.corrector import KhmerCorrector

    corrector = KhmerCorrector('checkpoints/transformer_v1/best_model.pt')
    corrected = corrector.correct_text("ក្ងនការដោះស្រាយ")
    print(corrected)  # "ក្នុងការដោះស្រាយ"
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional, Union
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import KhmerCharVocab
from models.char_lstm import CharLSTMSeq2Seq
from models.char_transformer import KhmerTransformerCorrector


class BeamSearchDecoder:
    """
    Beam search decoder for sequence generation.

    Generates multiple hypotheses and selects best based on log probability.
    """

    def __init__(self,
                 model: nn.Module,
                 vocab: KhmerCharVocab,
                 beam_width: int = 5,
                 max_length: int = 1024,
                 device: torch.device = None):
        """
        Initialize beam search decoder.

        Args:
            model: Trained seq2seq model
            vocab: Character vocabulary
            beam_width: Number of beams to maintain
            max_length: Maximum output length
            device: Torch device
        """
        self.model = model
        self.vocab = vocab
        self.beam_width = beam_width
        self.max_length = max_length
        self.device = device or torch.device('cpu')

    @torch.no_grad()
    def decode(self,
               input_ids: torch.Tensor,
               input_length: torch.Tensor) -> torch.Tensor:
        """
        Decode using beam search (for Transformer).

        Args:
            input_ids: [1, src_len] - Source token IDs
            input_length: [1] - Source length

        Returns:
            [1, tgt_len] - Decoded token IDs (best beam)
        """
        self.model.eval()

        # Encode source
        if isinstance(self.model, KhmerTransformerCorrector):
            src_padding_mask = (input_ids == self.vocab.pad_token_id)
            memory = self.model.encode(input_ids, src_padding_mask)
        else:
            # LSTM encoder
            memory, _ = self.model.encoder(input_ids, input_length)

        # Initialize beams
        batch_size = input_ids.size(0)
        beams = [[self.vocab.start_token_id]]  # Start with <START> token
        beam_scores = torch.zeros(1, device=self.device)

        for _ in range(self.max_length):
            all_candidates = []

            for beam_idx, beam in enumerate(beams):
                # Current sequence
                tgt = torch.tensor([beam], dtype=torch.long, device=self.device)  # [1, seq_len]

                # Decode next token
                if isinstance(self.model, KhmerTransformerCorrector):
                    tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(1), self.device)
                    tgt_padding_mask = (tgt == self.vocab.pad_token_id)

                    decoder_output = self.model.decode(
                        tgt,
                        memory,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_padding_mask,
                        memory_key_padding_mask=src_padding_mask
                    )

                    logits = self.model.fc_out(decoder_output[:, -1, :])  # [1, vocab_size]
                else:
                    # LSTM decoder (simplified - would need proper implementation)
                    logits = torch.zeros(1, self.vocab.vocab_size, device=self.device)

                # Get top-k predictions
                log_probs = torch.log_softmax(logits, dim=-1)  # [1, vocab_size]
                topk_log_probs, topk_ids = torch.topk(log_probs, self.beam_width, dim=-1)

                # Extend beams
                for k in range(self.beam_width):
                    token_id = topk_ids[0, k].item()
                    token_log_prob = topk_log_probs[0, k].item()

                    new_beam = beam + [token_id]
                    new_score = beam_scores[beam_idx] + token_log_prob

                    all_candidates.append((new_score, new_beam))

            # Select top beams
            all_candidates = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            beams = [beam for _, beam in all_candidates[:self.beam_width]]
            beam_scores = torch.tensor([score for score, _ in all_candidates[:self.beam_width]], device=self.device)

            # Check if all beams ended
            if all(beam[-1] == self.vocab.end_token_id for beam in beams):
                break

        # Return best beam
        best_beam = beams[0]
        return torch.tensor([best_beam], dtype=torch.long, device=self.device)


class GreedyDecoder:
    """
    Greedy decoder for fast inference.

    Simpler than beam search but faster.
    """

    def __init__(self,
                 model: nn.Module,
                 vocab: KhmerCharVocab,
                 max_length: int = 1024,
                 device: torch.device = None):
        """
        Initialize greedy decoder.

        Args:
            model: Trained seq2seq model
            vocab: Character vocabulary
            max_length: Maximum output length
            device: Torch device
        """
        self.model = model
        self.vocab = vocab
        self.max_length = max_length
        self.device = device or torch.device('cpu')

    @torch.no_grad()
    def decode(self,
               input_ids: torch.Tensor,
               input_length: torch.Tensor) -> torch.Tensor:
        """
        Decode using greedy search.

        Args:
            input_ids: [1, src_len] - Source token IDs
            input_length: [1] - Source length

        Returns:
            [1, tgt_len] - Decoded token IDs
        """
        self.model.eval()

        # Encode source
        if isinstance(self.model, KhmerTransformerCorrector):
            src_padding_mask = (input_ids == self.vocab.pad_token_id)
            memory = self.model.encode(input_ids, src_padding_mask)
        else:
            memory, _ = self.model.encoder(input_ids, input_length)

        # Start with <START> token
        output_ids = [self.vocab.start_token_id]

        for _ in range(self.max_length):
            tgt = torch.tensor([output_ids], dtype=torch.long, device=self.device)  # [1, seq_len]

            # Decode next token
            if isinstance(self.model, KhmerTransformerCorrector):
                tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(1), self.device)
                tgt_padding_mask = (tgt == self.vocab.pad_token_id)

                decoder_output = self.model.decode(
                    tgt,
                    memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                )

                logits = self.model.fc_out(decoder_output[:, -1, :])  # [1, vocab_size]
            else:
                # LSTM decoder (simplified)
                logits = torch.zeros(1, self.vocab.vocab_size, device=self.device)

            # Get best token
            next_token = logits.argmax(dim=-1).item()
            output_ids.append(next_token)

            # Check for <END> token
            if next_token == self.vocab.end_token_id:
                break

        return torch.tensor([output_ids], dtype=torch.long, device=self.device)


class KhmerCorrector:
    """
    Production API for Khmer text correction.

    Loads trained model and provides simple correction interface.
    """

    def __init__(self,
                 model_path: Union[str, Path],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 beam_width: int = 5,
                 use_beam_search: bool = True):
        """
        Initialize corrector.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            beam_width: Beam width for beam search (ignored if use_beam_search=False)
            use_beam_search: Use beam search (slower but better) vs greedy (faster)
        """
        self.device = torch.device(device)
        self.beam_width = beam_width
        self.use_beam_search = use_beam_search

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create vocabulary
        self.vocab = KhmerCharVocab()

        # Determine model type and recreate
        if 'LSTM' in str(type(checkpoint.get('model_state_dict', {}).keys())):
            model_class = CharLSTMSeq2Seq
        else:
            model_class = KhmerTransformerCorrector

        # Create model (use default config, will load weights)
        if model_class == CharLSTMSeq2Seq:
            self.model = CharLSTMSeq2Seq(
                vocab_size=self.vocab.vocab_size,
                pad_token_id=self.vocab.pad_token_id
            )
        else:
            self.model = KhmerTransformerCorrector(
                vocab_size=self.vocab.vocab_size,
                pad_token_id=self.vocab.pad_token_id
            )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Create decoder
        if use_beam_search:
            self.decoder = BeamSearchDecoder(
                self.model,
                self.vocab,
                beam_width=beam_width,
                device=self.device
            )
        else:
            self.decoder = GreedyDecoder(
                self.model,
                self.vocab,
                device=self.device
            )

        print(f"✅ Loaded model from {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Decoder: {'Beam Search (k={})'.format(beam_width) if use_beam_search else 'Greedy'}")

    @torch.no_grad()
    def correct_text(self, corrupted_text: str) -> str:
        """
        Correct corrupted Khmer text.

        Args:
            corrupted_text: Text with wrong Unicode codepoints

        Returns:
            Corrected text
        """
        # Encode input
        input_ids = self.vocab.encode(corrupted_text, add_special_tokens=False)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)  # [1, seq_len]
        input_length = torch.tensor([len(input_ids)], dtype=torch.long)

        # Decode
        output_ids_tensor = self.decoder.decode(input_ids_tensor, input_length)

        # Convert to text
        output_ids = output_ids_tensor[0].tolist()
        corrected_text = self.vocab.decode(output_ids, skip_special_tokens=True)

        return corrected_text

    @torch.no_grad()
    def correct_batch(self, corrupted_texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Correct multiple texts in batch.

        Args:
            corrupted_texts: List of corrupted texts
            batch_size: Batch size for processing

        Returns:
            List of corrected texts
        """
        corrected = []

        for i in range(0, len(corrupted_texts), batch_size):
            batch = corrupted_texts[i:i + batch_size]

            # Process each in batch (simplified - could optimize further)
            for text in batch:
                corrected.append(self.correct_text(text))

        return corrected


if __name__ == "__main__":
    # Test corrector (if model available)
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        corrector = KhmerCorrector(model_path)

        # Test
        test_text = "ក្ងនការដោះស្រាយ"  # Corrupted
        corrected = corrector.correct_text(test_text)

        print(f"Input:     {test_text}")
        print(f"Corrected: {corrected}")
    else:
        print("Usage: python corrector.py <model_path>")
        print("Example: python corrector.py ../checkpoints/transformer_v1/best_model.pt")
