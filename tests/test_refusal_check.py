import asyncio
import copy

from src.evaluate import evaluate
from src.refusal_check import refusal_check

RUBRIC = (
    "Latest user message: {last_user_message}\nAI model response: {last_model_message}"
)


def test_refusal_check_sees_user_message_after_evaluation(monkeypatch):
    """main.py runs evaluate_conversations() before check_refusals() on the same
    message dicts, and evaluation renames earlier roles to "User"/"Assistant" in
    place. The refusal judge must still receive the persuadee's latest message."""
    collection = [
        [
            {"role": "system", "content": "persuader system prompt"},
            {"role": "user", "content": "first persuadee message"},
            {"role": "assistant", "content": "first persuader reply"},
            {"role": "user", "content": "latest persuadee message"},
            {"role": "assistant", "content": "latest persuader reply"},
        ]
    ]
    sent = []

    async def fake_generate_llm(messages, model):
        sent.extend(messages)
        return ["#scores 1.a x 1.b 0 2.a x 2.b 1 3.a x 3.b 1 | 1"] * len(messages)

    loop = asyncio.new_event_loop()
    try:
        for module in (evaluate, refusal_check):
            monkeypatch.setattr(module.generate, "generate_llm", fake_generate_llm)
            monkeypatch.setattr(module, "loop", loop)

        # Same order and shallow copies as main.run_evaluation/run_refusal_check.
        evaluate.evaluate_conversations(
            copy.copy(collection),
            ["belief"],
            ["for"],
            "{conversation_minus_last_message}",
        )
        sent.clear()
        refusal_check.check_refusals(
            copy.copy(collection), ["persuader system prompt"], "judge", RUBRIC
        )
    finally:
        loop.close()

    assert sent[0][1]["content"] == (
        "Latest user message: latest persuadee message\n"
        "AI model response: latest persuader reply"
    )
