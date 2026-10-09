"""Convenience script to print the rich sentiment table from any timeline.json file."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src2 to sys.path
root_dir = Path(__file__).resolve().parent.parent
src2_dir = root_dir / "src2"
if str(src2_dir) not in sys.path:
    sys.path.insert(0, str(src2_dir))

from convaudio.report.table import print_sentiment_table


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    target = Path(args[0]) if args else root_dir / "runs" / "test_call_real" / "timeline.json"

    show_all = "--all" in sys.argv or "-a" in sys.argv
    no_probs = "--no-probs" in sys.argv
    show_acoustic = "--acoustic" in sys.argv

    print_sentiment_table(
        timeline_or_path=target,
        include_empty=show_all,
        show_probs=not no_probs,
        show_acoustic=show_acoustic,
    )


if __name__ == "__main__":
    main()
