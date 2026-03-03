from typing import List
import torch


class CTCGreedyDecoder:
    """
    Greedy CTC decoder - takes the most likely token at each timestep
    """
    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: The tokenizer used for encoding/decoding
        """
        self.tokenizer = tokenizer

    def decode(self,
               logits: torch.Tensor) -> List[int]:
        """
        Decode logits using greedy decoding - fully batched

        Args:
            logits: [batch_size, seq_len, vocab_size] raw logits

        Returns:
            List of decoded strings
        """
        # Get most likely tokens for entire batch
        predicted_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
        results = self.tokenizer.batch_decode(predicted_ids)
        return results