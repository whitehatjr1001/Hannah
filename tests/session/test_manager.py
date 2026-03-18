"""Session persistence tests for the nanobot-inspired Hannah chat store."""

from __future__ import annotations

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
