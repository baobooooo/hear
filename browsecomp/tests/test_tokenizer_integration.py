import pytest
pytest.importorskip("transformers", reason="Real tokenizer extra required")
pytest.importorskip("tokenizers", reason="Real tokenizer extra required")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from bcgraph.config import EndpointConfig
from bcgraph.tokenization import HuggingFaceCounter


def test_real_hf_chat_template_count_without_torch_or_downloads(tmp_path):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(tmp_path)
    counter = HuggingFaceCounter(EndpointConfig(tokenizer_path=str(tmp_path)))
    assert counter.exact
    assert counter.messages([{"role": "user", "content": "hello world hello world hello"}]) == 5




def test_request_thinking_override_counts_actual_template_without_mutation(tmp_path):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "hello {% if enable_thinking %}world{% endif %}"
    tokenizer.save_pretrained(tmp_path)
    counter = HuggingFaceCounter(EndpointConfig(tokenizer_path=str(tmp_path), enable_thinking=True))
    messages = [{"role": "user", "content": "test"}]
    assert counter.messages(messages, enable_thinking=False) == 1
    assert counter.messages(messages, enable_thinking=True) == 2
    assert counter.messages(messages) == 2


@pytest.mark.parametrize("engine,body,expected", [
    ("vllm", {"content": None}, 2), ("vllm", {}, 2),
    ("sparse-vllm", {"content": None}, 3),
])
def test_null_assistant_content_matches_serving_template_without_mutating_history(tmp_path, engine, body, expected):
    from copy import deepcopy
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(tmp_path)
    counter = HuggingFaceCounter(EndpointConfig(engine=engine, tokenizer_path=str(tmp_path)))
    messages = [{"role": "user", "content": "hello"},
                {"role": "assistant", "reasoning_content": "preserve reasoning", **body},
                {"role": "user", "content": "world"}]
    original = deepcopy(messages)
    assert counter.messages(messages, enable_thinking=False) == expected
    assert messages == original
