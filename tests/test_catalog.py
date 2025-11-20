from datetime import datetime
from pathlib import Path

from hangermon.storage import catalog


def test_clip_metadata_round_trip(tmp_path: Path):
    video = tmp_path / "2024/01/02"
    video.mkdir(parents=True)
    mp4 = video / "clip.mp4"
    mp4.write_bytes(b"fake")
    metadata = mp4.with_suffix(".mp4.json")
    metadata.write_text(
        """
        {
            "path": "clip.mp4",
            "timestamp": "2024-01-02T03:04:05",
            "duration": 2.5,
            "frames": 50,
            "confidence": 0.9
        }
        """.strip()
    )

    entries = catalog.list_clips(tmp_path)
    assert len(entries) == 1
    record = entries[0]
    assert record.path.name == "clip.mp4"
    assert record.duration == 2.5
    assert record.frames == 50
    assert record.confidence == 0.9
    assert isinstance(record.timestamp, datetime)
