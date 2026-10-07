"""Token accounting without modifying source text or assistant histories."""
from __future__ import annotations

import json
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol
from .config import EndpointConfig


class TokenCounter(Protocol):
    description: str
    exact: bool
    def text(self, value: str) -> int: ...
    def messages(self, messages: list[dict[str, Any]], *, enable_thinking: bool | None = None) -> int: ...


class Utf8Counter:
    """Explicitly approximate/conservative; tests and demos only by default."""
    description = "UTF-8 byte bound + message overhead (NOT model tokens)"
    exact = False

    @lru_cache(maxsize=512)
    def text(self, value: str) -> int:
        return len(value.encode("utf-8"))

    def messages(self, messages: list[dict[str, Any]], *, enable_thinking: bool | None = None) -> int:
        return 16 + sum(self.text(json.dumps(m, ensure_ascii=False)) + 16 for m in messages)


class HuggingFaceCounter:
    exact = True

    def __init__(self, config: EndpointConfig):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path or config.model,
            revision=config.tokenizer_revision,
            trust_remote_code=config.trust_remote_code,
        )
        self.encoder = None
        self.normalize_null_content = config.engine == "vllm"
        if config.tokenizer_format == "deepseek_v4":
            if not config.tokenizer_path:
                raise ValueError("DeepSeek V4 requires a local pinned tokenizer and official encoder")
            path = Path(config.tokenizer_path) / "encoding" / "encoding_dsv4.py"
            spec = importlib.util.spec_from_file_location("bcgraph_deepseek_v4_encoding", path)
            if spec is None or spec.loader is None:
                raise ValueError("Cannot load DeepSeek V4 reference encoder")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.encoder = module.encode_messages
            self.thinking_mode = "chat" if config.enable_thinking is False else "thinking"
        elif not self.tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat_template; use the serving model's exact template")
        self.kwargs = dict(config.chat_template_kwargs)
        if config.enable_thinking is not None:
            self.kwargs["enable_thinking"] = config.enable_thinking
        if config.preserve_thinking is not None:
            self.kwargs["preserve_thinking"] = config.preserve_thinking
        self.description = f"HF:{config.tokenizer_path or config.model}@{config.tokenizer_revision or 'local/default'}"
        if self.encoder:
            self.description += f":official-deepseek-v4:{self.thinking_mode}"

    @lru_cache(maxsize=512)
    def text(self, value: str) -> int:
        return len(self.tokenizer.encode(value, add_special_tokens=False))

    def messages(self, messages: list[dict[str, Any]], *, enable_thinking: bool | None = None) -> int:
        if self.encoder:
            mode = self.thinking_mode if enable_thinking is None else "thinking" if enable_thinking else "chat"
            prompt = self.encoder(messages, thinking_mode=mode)
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))
        kwargs = dict(self.kwargs)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        # vLLM renders null content as empty text. Sparse-vLLM retains None;
        # match the selected endpoint without changing the stored history.
        template_messages = [
            {**message, "content": ""}
            if self.normalize_null_content and message.get("content") is None else message
            for message in messages
        ]
        return len(self.tokenizer.apply_chat_template(
            template_messages, tokenize=True, return_dict=False, add_generation_prompt=True, **kwargs
        ))


def make_counter(config: EndpointConfig, allow_approximate: bool = False) -> TokenCounter:
    if config.tokenizer_path:
        return HuggingFaceCounter(config)
    if allow_approximate:
        return Utf8Counter()
    raise ValueError("Set tokenizer_path to the serving model's tokenizer; approximate counts are disabled")


def prefix_chars(text: str, limit: int, counter: TokenCounter) -> int:
    """Find a character prefix within a token cap, preserving exact source spans."""
    if limit <= 0:
        return 0
    # Probe a growing character prefix instead of tokenizing an entire very
    # long document before selecting a few thousand tokens. This preserves exact
    # source offsets and avoids huge tokenizer warnings/work for discarded text.
    high = min(len(text), max(128, limit))
    low = 0
    while high < len(text) and counter.text(text[:high]) <= limit:
        low = high
        high = min(len(text), high * 2)
    if counter.text(text[:high]) <= limit:
        return high
    while low < high:
        mid = (low + high + 1) // 2
        if counter.text(text[:mid]) <= limit:
            low = mid
        else:
            high = mid - 1
    # Token counts near BPE boundaries need not be strictly monotone. The final
    # check is authoritative; optimal packing is not required for safety.
    while low > 0 and counter.text(text[:low]) > limit:
        low -= 1
    return low
