from __future__ import annotations

import logging
from typing import Union

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from src.generate.constrain import StructuredEnforcer
from src.types.adapters import session_to_conversation
from src.types.session import Session
from src.types.adapters import Conversation

log = logging.getLogger(__name__)


class LLM:
    def __init__(
        self,
        model_name: str = "unsloth/Phi-4",
        chat_template: str = "phi-4",
        max_seq_length: int = 2048,
        device: str = "cuda",
    ):
        log.info(
            f"Initializing LLM model={model_name}, template={chat_template}, max_seq_length={max_seq_length}"
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        self.tokenizer = get_chat_template(tokenizer, chat_template)
        FastLanguageModel.for_inference(model)
        self.model = model
        self.device = device

    def generate(
        self,
        messages: Union[Conversation, Session],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        include_thoughts_in_prompt: bool = False,
        include_exec_results_in_prompt: bool = False,
    ) -> str:
        if isinstance(messages, Session):
            messages = session_to_conversation(
                messages,
                include_thoughts=include_thoughts_in_prompt,
                include_exec_results=include_exec_results_in_prompt,
            )

        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)

        n_input = inputs.shape[1]
        log.info(f"Completing {n_input} tokens...")

        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[StructuredEnforcer(self.tokenizer)],
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=temperature,
        )

        assert outputs.shape[0] == 1
        new_tokens = outputs[0, n_input:]
        log.info(f"Generated {new_tokens.shape[0]} tokens")

        return self.tokenizer.decode(new_tokens, skip_special_tokens=False)
