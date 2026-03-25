"""Session persistence tests for the nanobot-inspired Hannah chat store."""

from __future__ import annotations

from hannah.runtime.events import EventEnvelope
from hannah.session.event_records import serialize_event_record
from hannah.session.manager import SessionManager


def test_session_manager_persists_and_lists_sessions(tmp_path) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    session = manager.get_or_create("cli:test")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    manager.save(session)

    reloaded = SessionManager(sessions_dir=tmp_path).get_or_create("cli:test")

    assert [message["content"] for message in reloaded.messages] == ["hello", "world"]
    assert reloaded.get_recent(1) == [{"role": "assistant", "content": "world"}]

    listed = manager.list_sessions()
    assert len(listed) == 1
    assert listed[0]["key"] == "cli:test"
    assert listed[0]["message_count"] == 2


def test_event_records_use_a_stable_jsonl_shape() -> None:
    record = serialize_event_record(
        EventEnvelope.create(
            "subagent_completed",
            session_id="session-1",
            message_id="msg-1",
            worker_id="strategy",
            payload={"status": "completed"},
        ),
        session_id="session-1",
    )

    assert record["record_type"] == "event"
    assert record["session_id"] == "session-1"
    assert record["payload"]["event_type"] == "subagent_completed"


def test_session_manager_appends_event_records_without_promoting_them_to_messages(tmp_path) -> None:
    manager = SessionManager(sessions_dir=tmp_path)
    session = manager.get_or_create("cli:test")
    session.add_message("user", "hello")
    manager.save(session)

    manager.append_event(
        "cli:test",
        EventEnvelope.create(
            "subagent_completed",
            session_id="cli:test",
            message_id="msg-1",
            worker_id="strategy",
            payload={"status": "completed"},
        ),
    )

    reloaded = SessionManager(sessions_dir=tmp_path).get_or_create("cli:test")

    assert [message["content"] for message in reloaded.messages] == ["hello"]
    lines = (tmp_path / "cli_test.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"record_type": "event"' in line for line in lines)
