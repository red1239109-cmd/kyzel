from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple
import logging

from transformers import LogitsProcessor
from torch import Tensor, FloatTensor

logger = logging.getLogger(__name__)

# Verbatim start of a Python code block.
CODE_START = "```python\n"
CODE_START_LINE = "```python"


class State(Enum):
    START = auto()
    THOUGHT_CONTENT = auto()
    ACTION_OPEN = auto()
    ACTION_CONTENT = auto()
    CODE_FENCE_START = auto()
    CODE_CONTENT = auto()
    DONE = auto()


# -----------------------------
# 1) Logits forcing (mechanics)
# -----------------------------
def force_token(scores: FloatTensor, token_id: int) -> FloatTensor:
    """
    Enforce generation of `token_id`: set its score to 0 and all others to -inf.
    Assumes batch_size == 1.
    """
    if scores.dim() != 2:
        raise ValueError(f"expected scores [batch, vocab], got {tuple(scores.shape)}")
    if scores.size(0) != 1:
        raise ValueError(f"batch_size must be 1, got {scores.size(0)}")

    vocab = scores.size(-1)
    if not (0 <= token_id < vocab):
        raise ValueError(f"token_id {token_id} out of range [0, {vocab})")

    scores.fill_(float("-inf"))
    scores[0, token_id] = 0.0
    return scores


# ----------------------------------
# 2) Detection / parsing (read-only)
# ----------------------------------
def _split_lines(text: str) -> List[str]:
    return text.splitlines(keepends=False) if text else []


def prefix_before_code_fence(text: str) -> str:
    """
    Return only the part of `text` that occurs before the first ```python fence line.
    This prevents tag substring checks from triggering due to content inside code blocks.
    """
    lines = _split_lines(text)
    for i, line in enumerate(lines):
        # tolerant: ignore surrounding whitespace
        if line.strip() == CODE_START_LINE:
            return "\n".join(lines[:i])
    return text or ""


def code_block_status(text: str) -> Tuple[bool, bool]:
    """
    Markdown-aware code-block detector.

    Returns: (has_content, should_end)
      - has_content: whether the code block has any non-empty line inside it
      - should_end: whether a closing fence ``` was detected
    """
    if not text:
        return False, False

    lines_all = _split_lines(text)

    # Find exact start fence line (tolerant for whitespace).
    start_idx = None
    for i, line in enumerate(lines_all):
        if line.strip() == CODE_START_LINE:
            start_idx = i
            break
    if start_idx is None:
        return False, False

    lines = lines_all[start_idx + 1 :]

    for i, line in enumerate(lines):
        # closing fence must be a bare ``` (optionally followed by whitespace)
        if line.startswith("```") and not line[3:].strip():
            has_content = any(l.strip() for l in lines[:i])
            return has_content, True

    has_any_content = any(l.strip() for l in lines)
    return has_any_content, False


# ---------------------------
# 3) Policy (state decisions)
# ---------------------------
@dataclass
class Transition:
    new_state: State
    force_text: Optional[str] = None


def decide_next(state: State, generated_text: str) -> Transition:
    """
    Decide state transitions and forced insertions.

    IMPORTANT:
    - Tag detection only looks at prefix before the code fence (prevents injection via code content).
    """
    safe_text = prefix_before_code_fence(generated_text)

    if state == State.START:
        return Transition(State.THOUGHT_CONTENT, "<thought>")

    if state == State.THOUGHT_CONTENT and "</thought>" in safe_text:
        return Transition(State.ACTION_OPEN, "\n<action>")

    if state == State.ACTION_OPEN:
        return Transition(State.ACTION_CONTENT, None)

    if state == State.ACTION_CONTENT and "</action>" in safe_text:
        return Transition(State.CODE_FENCE_START, "\n" + CODE_START)

    if state == State.CODE_FENCE_START:
        return Transition(State.CODE_CONTENT, None)

    if state == State.CODE_CONTENT:
        has_content, should_end = code_block_status(generated_text)
        # Policy: require non-empty code blocks before finishing.
        if has_content and should_end:
            return Transition(State.DONE, "\n")

    return Transition(state, None)


# ----------------------------------------
# 4) Orchestration (ties everything together)
# ----------------------------------------
class StructuredEnforcer(LogitsProcessor):
    """
    Stateful LogitsProcessor that enforces a fixed structure:

      <thought> ... </thought>
      <action> ... </action>
      ```python
      ...
      ```
      (then EOS)

    Batch size is intentionally restricted to 1.
    """

    def __init__(self, tokenizer):
        if not hasattr(tokenizer, "eos_token_id") or tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must have a non-None eos_token_id")

        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

        self.state: State = State.START
        self.start_pos: Optional[int] = None

        # Pending forced token ids.
        self._force_queue: List[int] = []

        # Debug token history.
        self._token_history: List[Tuple[int, str]] = []

    def _pop_forced(self) -> Optional[int]:
        if not self._force_queue:
            return None
        return self._force_queue.pop(0)

    def _enqueue_force_text(self, text: str) -> None:
        # Encode without special tokens; we want raw token sequence insertion.
        self._force_queue = self.tokenizer.encode(text, add_special_tokens=False)

    def __call__(self, input_ids: Tensor, scores: FloatTensor) -> FloatTensor:
        # Safety: this processor is stateful per-sequence.
        if input_ids.size(0) != 1 or scores.size(0) != 1:
            raise ValueError(
                f"StructuredEnforcer supports batch_size=1 only. "
                f"got input_ids batch={input_ids.size(0)}, scores batch={scores.size(0)}"
            )

        if self.start_pos is None:
            self.start_pos = input_ids.shape[1]
            logger.debug(f"Generation starts at position {self.start_pos}")

        # Decode assistant-generated text (since start_pos).
        if generated_tokens := input_ids[0, self.start_pos:].tolist():
            new_token = generated_tokens[-1]
            self._log_new_token(new_token)
            generated_text = self.tokenizer.decode(generated_tokens)
        else:
            generated_text = ""

        while True:
            # Default: forbid EOS unless we deterministically finish.
            scores[0, self.eos_token_id] = float("-inf")

            # 1) Apply forced token if queued.
            tok = self._pop_forced()
            if tok is not None:  # IMPORTANT: token id can be 0
                logger.debug(
                    f"Forcing token_id={tok}, state={self.state}, queue_len={len(self._force_queue)}"
                )
                return force_token(scores, tok)

            # 2) Policy: decide if we should transition and/or enqueue a forced text.
            tr = decide_next(self.state, generated_text)
            if tr.new_state != self.state:
                logger.debug(f"State transition: {self.state} -> {tr.new_state}")
                self.state = tr.new_state
                if tr.force_text:
                    self._enqueue_force_text(tr.force_text)
                continue

            # 3) If DONE, force EOS.
            if self.state == State.DONE:
                logger.debug("State=DONE, forcing EOS")
                return force_token(scores, self.eos_token_id)

            # 4) Stable: allow model to proceed.
            return scores

    def _log_new_token(self, new_token: int) -> None:
        token_text = self.tokenizer.decode([new_token])
        self._token_history.append((new_token, token_text))
        logger.debug(f"Token {len(self._token_history)}: {new_token} -> {token_text!r}")

