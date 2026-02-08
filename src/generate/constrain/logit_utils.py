from torch import FloatTensor
import torch

def force_token(scores: FloatTensor, token_id: int) -> FloatTensor:
    """
    Enforce generation of `token_id`: set its score to 0 and all others to -inf.
    Supports batch processing and adds input validation.
    """
    # 1. Validation: Ensure input is [batch, vocab]
    if scores.dim() != 2:
        raise ValueError(f"expected 2D scores [batch, vocab], got shape={tuple(scores.shape)}")

    vocab = scores.size(-1)
    if not (0 <= token_id < vocab):
        raise ValueError(f"token_id {token_id} out of range [0, {vocab})")

    # 2. Logic: Apply forcing to the entire batch
    scores.fill_(float("-inf"))
    scores[:, token_id] = 0.0
    return scores
