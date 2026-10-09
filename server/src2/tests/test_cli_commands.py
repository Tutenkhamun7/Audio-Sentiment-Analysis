"""End-to-end CLI commands smoke test suite for Stage 5."""

import json
from pathlib import Path

from typer.testing import CliRunner

from convaudio.cli import app
from tests.fixtures.synth import generate_clean_stereo_call


def test_cli_all_commands_e2e(tmp_path: Path) -> None:
    wav_path = tmp_path / "call.wav"
    run_dir = tmp_path / "run"
    generate_clean_stereo_call(wav_path, duration_s=3.0)
    runner = CliRunner()

    # 1. run full pipeline
    r1 = runner.invoke(app, ["run", str(wav_path), "--out", str(run_dir), "--stub-models"])
    assert r1.exit_code == 0, f"run failed: {r1.output}"

    # 2. stage 5 individual run
    r2 = runner.invoke(app, ["stage", "5", "--run-dir", str(run_dir), "--stub-models"])
    assert r2.exit_code == 0, f"stage 5 failed: {r2.output}"

    # 3. validate timeline
    r3 = runner.invoke(app, ["validate", str(run_dir / "timeline.json")])
    assert r3.exit_code == 0, f"validate failed: {r3.output}"

    # 4. export-annotation
    annot_file_a = tmp_path / "annot_a.jsonl"
    r4 = runner.invoke(app, ["export-annotation", "--runs", str(run_dir), "--n", "10", "--out", str(annot_file_a)])
    assert r4.exit_code == 0, f"export-annotation failed: {r4.output}"
    assert annot_file_a.exists()

    # 5. Populate labels for agreement
    annot_file_b = tmp_path / "annot_b.jsonl"
    lines_a = [json.loads(line) for line in annot_file_a.read_text(encoding="utf-8").splitlines() if line.strip()]
    for item in lines_a:
        item["label"] = "neutral"
    annot_file_a.write_text("\n".join(json.dumps(x) for x in lines_a) + "\n", encoding="utf-8")
    annot_file_b.write_text("\n".join(json.dumps(x) for x in lines_a) + "\n", encoding="utf-8")

    r5 = runner.invoke(app, ["agreement", "--a", str(annot_file_a), "--b", str(annot_file_b)])
    assert r5.exit_code == 0, f"agreement failed: {r5.output}"

    # 6. weak-labels
    crm_file = tmp_path / "crm.csv"
    crm_file.write_text("call_id,escalated\ncall,true\n", encoding="utf-8")
    weak_out = tmp_path / "weak.jsonl"
    r6 = runner.invoke(app, ["weak-labels", "--crm", str(crm_file), "--runs", str(run_dir), "--out", str(weak_out)])
    assert r6.exit_code == 0, f"weak-labels failed: {r6.output}"
    assert weak_out.exists()

    # 7. train-semantic
    labels_file = tmp_path / "labels.jsonl"
    tl_str = str((run_dir / "timeline.json").resolve()).replace("\\", "/")
    tl_data = json.loads((run_dir / "timeline.json").read_text(encoding="utf-8"))
    turns = tl_data.get("turns", [])
    t0_id = turns[0]["turn_id"]
    t1_id = turns[1]["turn_id"] if len(turns) > 1 else "turn_001"
    records = [
        {"turn_id": t0_id, "timeline": tl_str, "label": "negative"},
        {"turn_id": t1_id, "timeline": tl_str, "label": "positive"},
    ]
    labels_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    model_out = tmp_path / "model_out"
    r7 = runner.invoke(app, ["train-semantic", "--labels", str(labels_file), "--out", str(model_out)])
    assert r7.exit_code == 0, f"train-semantic failed: {r7.output}"
    assert (model_out / "head_weights.npz").exists()
    assert (model_out / "artefact.json").exists()

    # 8. table command
    r8 = runner.invoke(app, ["table", str(run_dir)])
    assert r8.exit_code == 0, f"table failed: {r8.output}"
    assert "Time Range" in r8.output
    assert "Speaker" in r8.output
    assert "Semantic" in r8.output

    # 9. table command with options
    r9 = runner.invoke(app, ["table", str(run_dir / "timeline.json"), "--acoustic", "--no-probs"])
    assert r9.exit_code == 0, f"table options failed: {r9.output}"
    assert "Acoustic" in r9.output
