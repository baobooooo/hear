import json

import httpx
import pytest
import yaml

from bcgraph.demo import QUESTION, drive_nodes_for_test, make_demo_runtime
from bcgraph.prompts import MAIN_SYSTEM, MAIN_THINKING_SYSTEM
from bcgraph.schemas import message_text, parse_json_object


@pytest.mark.asyncio
@pytest.mark.parametrize("decision_mode", ["inherit", "off"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_main_thinking_plan_and_final_keep_reader_and_json_contract(store, enabled, decision_mode):
    runtime, backend, http = make_demo_runtime(store)
    runtime.config.main.enable_thinking = enabled
    runtime.config.workflow.decision_thinking = decision_mode
    runtime.config.workflow.planner_output_tokens = 4096 if enabled else 1400
    runtime.config.workflow.final_output_tokens = 8192 if enabled else 1400
    try:
        result = await drive_nodes_for_test(runtime, {
            "query_id": "demo", "question": QUESTION, "scope": "thinking", "attempt": 1})
        assert result["decision"]["exact_answer"] == "Aurora spectrograph"
        main = [r for r in backend.requests if r["messages"][-1]["content"].startswith("TASK:")]
        assert len(main) == 2
        for index, request in enumerate(main):
            expected = enabled if index == 0 or decision_mode == "inherit" else False
            assert request["chat_template_kwargs"]["enable_thinking"] is expected
            assert request["messages"][0]["content"] == (MAIN_THINKING_SYSTEM if enabled else MAIN_SYSTEM)
        assert [r["max_tokens"] for r in main] == ([4096, 8192] if enabled else [1400, 1400])
        reader = [r for r in backend.requests if r not in main]
        assert reader and all(r["chat_template_kwargs"]["enable_thinking"] is False for r in reader)
    finally:
        await runtime.close()
        await http.aclose()


@pytest.mark.asyncio
async def test_thinking_is_retained_in_history_but_never_parsed_as_answer(store):
    from bcgraph.config import EndpointConfig
    from bcgraph.transport import ChatClient

    hidden = '{"action":"answer","exact_answer":"not evidence"}'
    wire_reply = {"role": "assistant", "content": '{"action":"unresolved"}',
                  "reasoning_content": hidden}
    calls = []
    def backend(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": wire_reply, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 40}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client = ChatClient(EndpointConfig(enable_thinking=True), store, http=http)
        history = [{"role": "user", "content": "first"}]
        first = await client.complete(history, 8192, operation_id="first", writer_key="main")
        assert first["assistant"] == wire_reply
        assert parse_json_object(message_text(first["assistant"])) == {"action": "unresolved"}
        await client.complete([*history, first["assistant"], {"role": "user", "content": "next"}],
                              8192, operation_id="next", writer_key="main")
        assert calls[-1]["messages"][1]["reasoning_content"] == hidden
        with pytest.raises(ValueError, match="No JSON"):
            parse_json_object(message_text({"content": None, "reasoning_content": hidden}))








