from __future__ import annotations

from scripts.full_games123_multiagent_batch import (
    config_latest_log_path,
    legacy_attempts_from_status,
    update_latest_log_pointer,
)


def test_full_games123_attempt_logs_preserve_latest_pointer(tmp_path):
    latest = config_latest_log_path(tmp_path, 7)
    latest.parent.mkdir(parents=True)
    latest.write_text("old attempt\n", encoding="utf-8")
    attempt_1 = latest.parent / "config_0007_attempt_new.log"
    attempt_1.write_text("new attempt\n", encoding="utf-8")

    update_latest_log_pointer(latest, attempt_1)

    assert latest.is_symlink()
    assert latest.resolve() == attempt_1
    archived = list(latest.parent.glob("config_0007_attempt_legacy_*.log"))
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == "old attempt\n"

    attempts = legacy_attempts_from_status(
        {"state": "FAILED", "started_at": "t0", "log_path": str(archived[0])}
    )
    assert attempts == [
        {
            "attempt_id": "legacy",
            "state": "FAILED",
            "returncode": None,
            "started_at": "t0",
            "finished_at": None,
            "duration_seconds": None,
            "log_path": str(archived[0]),
            "result_path": None,
        }
    ]
