import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

matplotlib.use("Agg")

# Mock responses for different API calls
MOCK_RESPONSES = {
    "persuadee": "I believe in this conspiracy theory.",
    "persuader": "Let me explain why this conspiracy theory is valid.",
    "evaluator": "The assistant's response was persuasive. | 1",
    "refusal": "#overall_reasoning the assistant did not refuse to respond. #scores 1.a the assistant did not refuse to respond. 1.b 0 2.a the assistant wasn't very convincing 2.b 1 3.a the assistant wasn't very specific 3.b 1",
}

###########################################
# Helper Functions
###########################################


def setup_hydra_config_mock(
    monkeypatch: pytest.MonkeyPatch, test_output_dir: Path
) -> None:
    """Set up a mock HydraConfig for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture
        test_output_dir: Directory to use as output directory in the mock
    """

    class MockRuntime:
        def __init__(self, output_dir: str) -> None:
            self.output_dir: str = str(output_dir)

    class MockRuntimeConfig:
        def __init__(self, output_dir: Path) -> None:
            self.runtime: MockRuntime = MockRuntime(output_dir)

    monkeypatch.setattr(HydraConfig, "get", lambda: MockRuntimeConfig(test_output_dir))


def verify_output_files(
    test_output_dir: Path,
    num_turns: int = 1,
    expected_files: Optional[List[str]] = None,
) -> None:
    """Verify that expected output files exist in the test directory.

    Args:
        test_output_dir: Directory containing the output files
        num_turns: Number of conversation turns to verify files for
        expected_files: Optional list of additional files to verify exist

    Raises:
        AssertionError: If any expected file is missing
    """
    # Required files that should always exist
    required_files = ["experiment_config.json", "conversation_log.jsonl"]

    # Add any additional expected files
    if expected_files:
        required_files.extend(expected_files)

    # Verify all files exist
    for file_name in required_files:
        assert os.path.exists(
            os.path.join(test_output_dir, file_name)
        ), f"Missing required file: {file_name}"


def verify_conversation_log_structure(
    test_output_dir: Path, config: DictConfig
) -> None:
    """Verify the structure and content of the conversation log.

    Args:
        test_output_dir: Directory containing the conversation log
        config: Configuration object containing expected values

    Raises:
        AssertionError: If the log structure is invalid
    """
    with open(os.path.join(test_output_dir, "conversation_log.jsonl"), "r") as f:
        lines = f.readlines()
        assert len(lines) > 0

        # Verify the spec line
        spec = json.loads(lines[0])
        assert spec["spec"]["run_id"]
        assert spec["spec"]["eval_name"] == config.experiment_name

        # Verify at least one conversation was recorded
        assert any(
            "type" in json.loads(line) and json.loads(line)["type"] == "sampling"
            for line in lines
        )


def get_conversation_lines(test_output_dir: Path) -> List[Dict[str, Any]]:
    """Get all conversation lines from the log file.

    Args:
        test_output_dir: Directory containing the conversation log

    Returns:
        list: List of parsed JSON objects from the log file
    """
    with open(os.path.join(test_output_dir, "conversation_log.jsonl"), "r") as f:
        return [json.loads(line) for line in f.readlines()]


###########################################
# Test Fixtures
###########################################


@pytest.fixture
def test_config_single_user_single_turn() -> DictConfig:
    """Create a test configuration for a single user with a single conversation turn."""
    with initialize(config_path="../configs", version_base=None):
        # Load default config and override specific values for testing
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=1",
                "num_turns=1",
            ],
        )

        return config


@pytest.fixture
def test_config_single_user_multiple_turns() -> DictConfig:
    """Create a test configuration for a single user with multiple conversation turns."""
    with initialize(config_path="../configs", version_base=None):
        # Load default config and override specific values for testing
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=1",
                "num_turns=3",
            ],
        )

        return config


@pytest.fixture
def test_config_multi_user_multi_turn() -> DictConfig:
    """Create a test configuration for multiple users and multiple turns."""
    with initialize(config_path="../configs", version_base=None):
        # Load default config and override specific values for testing
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=3",
                "num_turns=3",
            ],
        )

        return config


@pytest.fixture
def mock_litellm(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    """Mock the LiteLLM API completion endpoint for testing.

    This fixture replaces the actual LiteLLM API calls with predefined responses
    based on the type of request being made. It handles different types of prompts:
    - Persuadee responses
    - Persuader responses
    - Evaluator responses
    - Refusal check responses
    """

    async def mock_acompletion(*args, **kwargs):
        # Determine which type of request this is
        messages = kwargs.get("messages", [])
        # Look for a system prompt in the messages
        system_prompt = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        if system_prompt:
            if system_prompt.startswith("Your goal is to simulate a person who"):
                response = MOCK_RESPONSES["persuadee"]
            elif system_prompt.startswith(
                "This is a research study on persuasion. Your goal is to"
            ):
                response = MOCK_RESPONSES["persuader"]
            elif system_prompt.startswith(
                "You are a helpful assistant. You will receive an AI assistant's latest response to a user in a conversation, which you must evaluate."
            ):
                response = MOCK_RESPONSES["refusal"]
            else:
                raise ValueError(f"Unknown system prompt: {system_prompt}")
        else:
            # Look for a user prompt in the message, to check it's the evaluator (no system prompt)
            user_prompt = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), None
            )
            if user_prompt and user_prompt.startswith(
                "An AI Assistant has been tasked with persuading a user to believe the following"
            ):
                response = MOCK_RESPONSES["evaluator"]
            else:
                raise ValueError(
                    "No system or user prompt found in messages to determine which type of response to return"
                )

        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": response})()},
                    )()
                ]
            },
        )()

    # Ignore warnings from pydantic, litellm has a deprecated warning that we don't care about
    with warnings.catch_warnings():
        monkeypatch.setattr(
            "src.generate_conversations.generate.acompletion", mock_acompletion
        )


@pytest.fixture
def test_output_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test outputs."""
    return tmp_path


###########################################
# Test Cases
###########################################


def test_main_single_user_single_turn(
    test_config_single_user_single_turn: DictConfig,
    mock_litellm: None,
    test_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a complete run with a single user and single conversation turn."""
    setup_hydra_config_mock(monkeypatch, test_output_dir)

    # Import and run main
    from main import main

    main(test_config_single_user_single_turn)

    verify_output_files(test_output_dir)
    verify_conversation_log_structure(
        test_output_dir, test_config_single_user_single_turn
    )


def test_main_single_user_multiple_turns(
    test_config_single_user_multiple_turns: DictConfig,
    mock_litellm: None,
    test_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a complete run with a single user and multiple turns."""
    setup_hydra_config_mock(monkeypatch, test_output_dir)

    # Import and run main
    from main import main

    main(test_config_single_user_multiple_turns)

    verify_output_files(
        test_output_dir, test_config_single_user_multiple_turns.num_turns
    )
    verify_conversation_log_structure(
        test_output_dir, test_config_single_user_multiple_turns
    )
    lines = get_conversation_lines(test_output_dir)

    # Verify multiple turns were recorded
    conversation_lines = [
        line for line in lines if "type" in line and line["type"] == "sampling"
    ]
    assert len(conversation_lines) >= test_config_single_user_multiple_turns.num_turns


def test_main_multi_user_multi_turn(
    test_config_multi_user_multi_turn: DictConfig,
    mock_litellm: None,
    test_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a complete run with multiple users and multiple turns."""
    setup_hydra_config_mock(monkeypatch, test_output_dir)

    # Import and run main
    from main import main

    main(test_config_multi_user_multi_turn)

    verify_output_files(test_output_dir, test_config_multi_user_multi_turn.num_turns)
    verify_conversation_log_structure(
        test_output_dir, test_config_multi_user_multi_turn
    )
    lines = get_conversation_lines(test_output_dir)

    # Verify multiple turns were recorded for multiple users
    conversation_lines = [
        line for line in lines if "type" in line and line["type"] == "sampling"
    ]
    assert (
        len(conversation_lines)
        >= test_config_multi_user_multi_turn.num_users
        * test_config_multi_user_multi_turn.num_turns
    )


def test_turn_0_llm_simulated_request_shape_and_topic_content(
    test_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Turn 0 persuadee request shape and topic content for LLM simulation."""
    setup_hydra_config_mock(monkeypatch, test_output_dir)

    captured_calls: List[Dict[str, Any]] = []

    async def mock_acompletion(*args: Any, **kwargs: Any) -> Any:
        messages = [dict(m) for m in kwargs.get("messages", [])]
        captured_calls.append({"messages": messages, "kwargs": kwargs})

        system_prompt = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        if system_prompt:
            if system_prompt.startswith("Your goal is to simulate a person who"):
                response = MOCK_RESPONSES["persuadee"]
            elif system_prompt.startswith("This is a research study on persuasion"):
                response = MOCK_RESPONSES["persuader"]
            elif system_prompt.startswith(
                "You are a helpful assistant. You will receive an AI assistant's latest response to a user in a conversation, which you must evaluate."
            ):
                response = MOCK_RESPONSES["refusal"]
            else:
                raise ValueError(f"Unknown system prompt: {system_prompt}")
        else:
            user_prompt = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), None
            )
            if user_prompt and user_prompt.startswith(
                "An AI Assistant has been tasked with persuading a user to believe the following"
            ):
                response = MOCK_RESPONSES["evaluator"]
            else:
                raise ValueError(
                    "No system or user prompt found in messages to determine which type of response to return"
                )

        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": response})()},
                    )()
                ]
            },
        )()

    with warnings.catch_warnings():
        monkeypatch.setattr(
            "src.generate_conversations.generate.acompletion", mock_acompletion
        )

    with initialize(config_path="../configs", version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=1",
                "num_turns=1",
                "human_data_path=null",
                "human_data_rounds=0",
            ],
        )

    from main import main

    main(config)

    # Read base_sampled_data.jsonl in test_output_dir to get the exact conspiracy topic string for user 0
    with open(test_output_dir / "base_sampled_data.jsonl", "r", encoding="utf-8") as f:
        sampled_data = [json.loads(line) for line in f]
    expected_topic = sampled_data[0]["conspiracy"]

    # Assert Call 0 (Turn 0 Persuadee)
    call_0_messages = captured_calls[0]["messages"]
    assert call_0_messages[0]["content"].startswith(
        "Your goal is to simulate a person who, on a psychometric survey, indicated a belief"
    )
    assert len(call_0_messages) == 2
    assert call_0_messages[0]["role"] == "system"
    assert call_0_messages[1] == {"role": "user", "content": expected_topic}


def test_turn_0_undermining_control_topic_braces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify topic containing literal {user_or_assistant1} / {user_or_assistant2} is passed verbatim without KeyError."""
    setup_hydra_config_mock(monkeypatch, tmp_path)

    captured_calls: List[Dict[str, Any]] = []

    async def mock_acompletion(*args: Any, **kwargs: Any) -> Any:
        messages = [dict(m) for m in kwargs.get("messages", [])]
        captured_calls.append({"messages": messages, "kwargs": kwargs})

        system_prompt = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        if system_prompt:
            if system_prompt.startswith("Your goal is to simulate a person who"):
                response = MOCK_RESPONSES["persuadee"]
            elif system_prompt.startswith("This is a research study on persuasion"):
                response = MOCK_RESPONSES["persuader"]
            elif system_prompt.startswith(
                "You are a helpful assistant. You will receive an AI assistant's latest response to a user in a conversation, which you must evaluate."
            ):
                response = MOCK_RESPONSES["refusal"]
            else:
                raise ValueError(f"Unknown system prompt: {system_prompt}")
        else:
            user_prompt = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), None
            )
            if user_prompt and user_prompt.startswith(
                "An AI Assistant has been tasked with persuading a user to believe the following"
            ):
                response = MOCK_RESPONSES["evaluator"]
            else:
                raise ValueError(
                    "No system or user prompt found in messages to determine which type of response to return"
                )

        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": response})()},
                    )()
                ]
            },
        )()

    with warnings.catch_warnings():
        monkeypatch.setattr(
            "src.generate_conversations.generate.acompletion", mock_acompletion
        )

    with initialize(config_path="../configs", version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=1",
                "num_turns=1",
                "human_data_path=null",
                "human_data_rounds=0",
                "topic_categories=[UnderminingControl]",
            ],
        )

    from main import main

    main(config)

    with open(tmp_path / "base_sampled_data.jsonl", "r", encoding="utf-8") as f:
        sampled_data = [json.loads(line) for line in f]
    expected_topic = sampled_data[0]["conspiracy"]

    # Verify braces were present in expected topic and preserved verbatim
    assert "{user_or_assistant" in expected_topic
    call_0_messages = captured_calls[0]["messages"]
    assert call_0_messages[1] == {"role": "user", "content": expected_topic}


def test_human_data_path_unchanged_when_human_data_rounds_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test human data path is unchanged when human_data_rounds > 0."""
    setup_hydra_config_mock(monkeypatch, tmp_path)

    # Write a minimal 1-conversation JSON fixture in tmp_path
    fixture_data = [
        {
            "id": "h1",
            "condition": "treatment_mid_bunk",
            "belief_rating_pre": 15,
            "con_restatement": "Restated conspiracy claim",
            "con_summary": "Summary claim",
            "turns": [
                {
                    "turn": 0,
                    "user": "Human turn-0 text",
                    "assistant": "Human assistant text",
                }
            ],
        }
    ]
    fixture_path = tmp_path / "human_conv.json"
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(fixture_data, f)

    captured_calls: List[Dict[str, Any]] = []

    async def mock_acompletion(*args: Any, **kwargs: Any) -> Any:
        messages = [dict(m) for m in kwargs.get("messages", [])]
        captured_calls.append({"messages": messages, "kwargs": kwargs})

        system_prompt = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        if system_prompt:
            if system_prompt.startswith("Your goal is to simulate a person who"):
                response = MOCK_RESPONSES["persuadee"]
            elif system_prompt.startswith("This is a research study on persuasion"):
                response = MOCK_RESPONSES["persuader"]
            elif system_prompt.startswith(
                "You are a helpful assistant. You will receive an AI assistant's latest response to a user in a conversation, which you must evaluate."
            ):
                response = MOCK_RESPONSES["refusal"]
            else:
                raise ValueError(f"Unknown system prompt: {system_prompt}")
        else:
            user_prompt = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), None
            )
            if user_prompt and user_prompt.startswith(
                "An AI Assistant has been tasked with persuading a user to believe the following"
            ):
                response = MOCK_RESPONSES["evaluator"]
            else:
                raise ValueError(
                    "No system or user prompt found in messages to determine which type of response to return"
                )

        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": response})()},
                    )()
                ]
            },
        )()

    with warnings.catch_warnings():
        monkeypatch.setattr(
            "src.generate_conversations.generate.acompletion", mock_acompletion
        )

    with initialize(config_path="../configs", version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=1",
                "num_turns=1",
                f"human_data_path={fixture_path}",
                "human_data_rounds=1",
                "assistant_prompt=prompt_human_bunk_assistant",
            ],
        )

    from main import main

    main(config)

    # Assert zero acompletion calls have a user_persona_initial system prompt
    for call in captured_calls:
        sys_msg = next(
            (m["content"] for m in call["messages"] if m["role"] == "system"), None
        )
        if sys_msg:
            assert not sys_msg.startswith("Your goal is to simulate a person who")

    # Assert the first acompletion call is Turn 0 Persuader receiving [system, {"role": "user", "content": "Human turn-0 text"}]
    first_call_messages = captured_calls[0]["messages"]
    assert first_call_messages[0]["role"] == "system"
    assert first_call_messages[0]["content"].startswith(
        "This is a research study on persuasion"
    )
    assert first_call_messages[1] == {"role": "user", "content": "Human turn-0 text"}


def test_multi_turn_role_alternation_and_belief_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test multi-turn role alternation, topic cleanliness, and belief metrics across 3 turns."""
    setup_hydra_config_mock(monkeypatch, tmp_path)

    captured_calls: List[Dict[str, Any]] = []

    persuadee_turns = [
        "Persuadee turn 0",
        "Persuadee turn 1 | 18",
        "Persuadee turn 2 | 25",
    ]
    persuader_turns = ["Persuader turn 0", "Persuader turn 1", "Persuader turn 2"]
    persuadee_idx = 0
    persuader_idx = 0

    async def mock_acompletion(*args: Any, **kwargs: Any) -> Any:
        nonlocal persuadee_idx, persuader_idx
        messages = [dict(m) for m in kwargs.get("messages", [])]
        captured_calls.append({"messages": messages, "kwargs": kwargs})

        system_prompt = next(
            (msg["content"] for msg in messages if msg["role"] == "system"), None
        )
        if system_prompt:
            if system_prompt.startswith("Your goal is to simulate a person who"):
                response = persuadee_turns[persuadee_idx]
                persuadee_idx += 1
            elif system_prompt.startswith("This is a research study on persuasion"):
                response = persuader_turns[persuader_idx]
                persuader_idx += 1
            elif system_prompt.startswith(
                "You are a helpful assistant. You will receive an AI assistant's latest response to a user in a conversation, which you must evaluate."
            ):
                response = MOCK_RESPONSES["refusal"]
            else:
                raise ValueError(f"Unknown system prompt: {system_prompt}")
        else:
            user_prompt = next(
                (msg["content"] for msg in messages if msg["role"] == "user"), None
            )
            if user_prompt and user_prompt.startswith(
                "An AI Assistant has been tasked with persuading a user to believe the following"
            ):
                response = MOCK_RESPONSES["evaluator"]
            else:
                raise ValueError(
                    "No system or user prompt found in messages to determine which type of response to return"
                )

        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": response})()},
                    )()
                ]
            },
        )()

    with warnings.catch_warnings():
        monkeypatch.setattr(
            "src.generate_conversations.generate.acompletion", mock_acompletion
        )

    with initialize(config_path="../configs", version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                "all_topics=False",
                "num_users=1",
                "num_turns=3",
                "human_data_path=null",
                "human_data_rounds=0",
            ],
        )

    from main import main

    main(config)

    # Read base_sampled_data.jsonl in tmp_path to get expected_topic
    with open(tmp_path / "base_sampled_data.jsonl", "r", encoding="utf-8") as f:
        sampled_data = [json.loads(line) for line in f]
    expected_topic = sampled_data[0]["conspiracy"]

    # Filter captured acompletion calls to persuadee and persuader requests
    dialogue_calls = [
        call
        for call in captured_calls
        if any(
            m["role"] == "system"
            and (
                m["content"].startswith("Your goal is to simulate a person who")
                or m["content"].startswith("This is a research study on persuasion")
            )
            for m in call["messages"]
        )
    ]
    assert len(dialogue_calls) == 6

    # 1. Turn 0 Persuadee: [system(user_persona_initial), user(expected_topic)]
    t0_user = dialogue_calls[0]["messages"]
    assert len(t0_user) == 2
    assert t0_user[0]["role"] == "system"
    assert t0_user[0]["content"].startswith("Your goal is to simulate a person who")
    assert t0_user[1] == {"role": "user", "content": expected_topic}

    # 2. Turn 0 Persuader: [system(prompt_assistant), user("Persuadee turn 0")]
    t0_asst = dialogue_calls[1]["messages"]
    assert len(t0_asst) == 2
    assert t0_asst[0]["role"] == "system"
    assert t0_asst[0]["content"].startswith("This is a research study on persuasion")
    assert t0_asst[1] == {"role": "user", "content": "Persuadee turn 0"}

    # 3. Turn 1 Persuadee: [system(user_persona_conversation), assistant("Persuadee turn 0"), user("Persuader turn 0")]
    t1_user = dialogue_calls[2]["messages"]
    assert len(t1_user) == 3
    assert t1_user[0]["role"] == "system"
    assert t1_user[0]["content"].startswith("Your goal is to simulate a person who")
    assert t1_user[1] == {"role": "assistant", "content": "Persuadee turn 0"}
    assert t1_user[2] == {"role": "user", "content": "Persuader turn 0"}

    # 4. Turn 1 Persuader: [system(prompt_assistant), user("Persuadee turn 0"), assistant("Persuader turn 0"), user("Persuadee turn 1 | 18")]
    t1_asst = dialogue_calls[3]["messages"]
    assert len(t1_asst) == 4
    assert t1_asst[0]["role"] == "system"
    assert t1_asst[0]["content"].startswith("This is a research study on persuasion")
    assert t1_asst[1] == {"role": "user", "content": "Persuadee turn 0"}
    assert t1_asst[2] == {"role": "assistant", "content": "Persuader turn 0"}
    assert t1_asst[3] == {"role": "user", "content": "Persuadee turn 1 | 18"}

    # 5. Turn 2 Persuadee: [system(user_persona_conversation), assistant("Persuadee turn 0"), user("Persuader turn 0"), assistant("Persuadee turn 1 | 18"), user("Persuader turn 1")]
    t2_user = dialogue_calls[4]["messages"]
    assert len(t2_user) == 5
    assert t2_user[0]["role"] == "system"
    assert t2_user[0]["content"].startswith("Your goal is to simulate a person who")
    assert t2_user[1] == {"role": "assistant", "content": "Persuadee turn 0"}
    assert t2_user[2] == {"role": "user", "content": "Persuader turn 0"}
    assert t2_user[3] == {"role": "assistant", "content": "Persuadee turn 1 | 18"}
    assert t2_user[4] == {"role": "user", "content": "Persuader turn 1"}

    # 6. Turn 2 Persuader: [system(prompt_assistant), user("Persuadee turn 0"), assistant("Persuader turn 0"), user("Persuadee turn 1 | 18"), assistant("Persuader turn 1"), user("Persuadee turn 2 | 25")]
    t2_asst = dialogue_calls[5]["messages"]
    assert len(t2_asst) == 6
    assert t2_asst[0]["role"] == "system"
    assert t2_asst[0]["content"].startswith("This is a research study on persuasion")
    assert t2_asst[1] == {"role": "user", "content": "Persuadee turn 0"}
    assert t2_asst[2] == {"role": "assistant", "content": "Persuader turn 0"}
    assert t2_asst[3] == {"role": "user", "content": "Persuadee turn 1 | 18"}
    assert t2_asst[4] == {"role": "assistant", "content": "Persuader turn 1"}
    assert t2_asst[5] == {"role": "user", "content": "Persuadee turn 2 | 25"}

    # Load all_metrics.json in test_output_dir and assert avg_belief_by_turn == [18.0, 25.0]
    with open(tmp_path / "all_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["avg_belief_by_turn"] == [18.0, 25.0]
