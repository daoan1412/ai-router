from __future__ import annotations

from ai_router.openai_api import (
    build_completion,
    build_models,
    error_body,
    messages_to_prompt,
    resolve_model,
    resolve_target,
    wants_reasoning,
)


def test_resolve_target_exact_openrouter_ids():
    t = resolve_target("qwen/qwen3.8-max-0902")
    assert t.provider == "qwen" and t.model_choice == "Max" and t.deep_think is False
    t = resolve_target("qwen/qwen3.8-omni-flash")
    assert t.model_choice == "Flash"
    t = resolve_target("qwen/qwen3.7-plus")
    assert t.model_choice == "Plus"
    t = resolve_target("deepseek/deepseek-r1-0528")
    assert t.provider == "deepseek" and t.deep_think is True
    t = resolve_target("deepseek/deepseek-chat")
    assert t.deep_think is False


def test_resolve_target_family_fallback():
    # Any qwen/* slug resolves to the nearest web model by keyword.
    assert resolve_target("qwen/qwen3-max").model_choice == "Max"
    assert resolve_target("qwen/qwen3-coder-flash").model_choice == "Flash"
    assert resolve_target("qwen/qwen-plus").model_choice == "Plus"
    assert resolve_target("qwen/qwen2.5-72b-instruct").model_choice == "Plus"
    # DeepSeek reasoning keyed on r1/reason/think.
    assert resolve_target("deepseek/deepseek-v3.2").deep_think is False
    assert resolve_target("deepseek/deepseek-r1").deep_think is True
    assert resolve_target("gpt-4o") is None
    assert resolve_target(None) is None


def test_resolve_target_zai():
    # Exact OpenRouter ids map to the web picker labels.
    assert resolve_target("z-ai/glm-5.3-flash").model_choice == "GLM-5.3-Flash"
    assert resolve_target("z-ai/glm-5.3").model_choice == "GLM-5.3"
    assert resolve_target("z-ai/glm-5.2").model_choice == "GLM-5.2"
    assert resolve_target("z-ai/glm-5.3").provider == "zai"
    # Other z-ai/* slugs fall back by keyword.
    assert resolve_target("z-ai/glm-4.7-flash").model_choice == "GLM-5.3-Flash"
    assert resolve_target("z-ai/glm-4.5-air").model_choice == "GLM-5.3-Flash"
    assert resolve_target("z-ai/glm-5.3-prime").model_choice == "GLM-5.3"
    assert resolve_target("~z-ai/glm-latest") is None  # OpenRouter alias syntax not supported
    assert resolve_target("glm-5.2").model_choice == "GLM-5.2"


def test_resolve_model_and_wants_reasoning_wrappers():
    assert resolve_model("qwen/qwen3.8-max-0902") == "qwen"
    assert resolve_model("deepseek/deepseek-chat") == "deepseek"
    assert resolve_model("gpt-4o") is None
    assert wants_reasoning("deepseek/deepseek-r1-0528") is True
    assert wants_reasoning("deepseek/deepseek-chat") is False
    assert wants_reasoning("qwen/qwen3.7-plus") is False
    assert wants_reasoning(None) is False


def test_messages_single_user_passthrough():
    assert messages_to_prompt([{"role": "user", "content": "2+2?"}]) == "2+2?"


def test_messages_system_plus_user():
    out = messages_to_prompt(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
        ]
    )
    assert out == "Be terse.\n\nHi"


def test_messages_multiturn_labelled():
    out = messages_to_prompt(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
        ]
    )
    assert out == "User: A\n\nAssistant: B\n\nUser: C"


def test_messages_content_parts():
    out = messages_to_prompt(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url"}]}]
    )
    assert out == "hello"


def test_build_completion_with_reasoning():
    resp = build_completion("deepseek/deepseek-r1-0528", "4", "2+2 is 4 because...")
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "4"
    assert msg["reasoning_content"] == "2+2 is 4 because..."
    assert resp["object"] == "chat.completion"
    assert resp["model"] == "deepseek/deepseek-r1-0528"


def test_build_completion_without_reasoning_omits_field():
    resp = build_completion("qwen/qwen3.7-plus", "hi", None)
    assert "reasoning_content" not in resp["choices"][0]["message"]


def test_build_models_lists_openrouter_ids():
    ids = {m["id"] for m in build_models()["data"]}
    assert "deepseek/deepseek-chat" in ids
    assert "qwen/qwen3.8-max-0902" in ids
    assert all("/" in i for i in ids)


def test_error_body_shape():
    body = error_body("bad", code="x")
    assert body["error"]["message"] == "bad"
    assert body["error"]["code"] == "x"
