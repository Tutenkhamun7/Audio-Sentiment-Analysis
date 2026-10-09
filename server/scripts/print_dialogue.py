"""Convenience script to print dialogue directly from any timeline.json file."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src2 to sys.path so convaudio can be imported directly
root_dir = Path(__file__).resolve().parent.parent
src2_dir = root_dir / "src2"
if str(src2_dir) not in sys.path:
    sys.path.insert(0, str(src2_dir))

from convaudio.report.dialogue import print_dialogue


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    target = Path(args[0]) if args else root_dir / "runs" / "test_call_no_split" / "timeline.json"

    merge = "--merge" in sys.argv or "-m" in sys.argv
    timestamps = "--timestamps" in sys.argv or "-t" in sys.argv

    print_dialogue(
        target,
        humanize_speakers=True,
        merge_consecutive=merge,
        include_timestamps=timestamps,
    )


if __name__ == "__main__":
    main()
