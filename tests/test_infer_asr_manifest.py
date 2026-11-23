"""Tests for ASR manifest encoding validation and error guidance (#3177)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import infer
from tests.conftest import strip_ansi


@pytest.fixture(autouse=True)
def isolated_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        infer,
        "console",
        Console(width=250, height=40, force_terminal=True, color_system="standard", no_color=False),
    )


def test_manifest_utf16_refused_with_guidance(monkeypatch):
    manifest = Path("manifest.jsonl")
    manifest.write_text('{"audio": "clip.wav"}\n', encoding="utf-16")
    output = Path("output.jsonl")

    res = CliRunner().invoke(app, [
        "infer", "--task", "asr", "--model", "openai/whisper-tiny", "--device", "cpu",
        "--input", str(manifest), "--output", str(output),
    ])
    assert res.exit_code == 1
    clean = " ".join(strip_ansi(res.output).split())
    assert "Cannot read manifest.jsonl as UTF-8. Save the ASR manifest as UTF-8 and retry." in clean
    assert "UnicodeDecodeError" not in clean
    assert not output.exists()


def test_manifest_invalid_byte_mid_file_refused_with_guidance(monkeypatch):
    manifest = Path("manifest_mid_error.jsonl")
    # First row valid UTF-8, second row contains invalid UTF-8 start byte
    content = b'{"audio": "clip1.wav"}\n\xff\xfe{"audio": "clip2.wav"}\n'
    manifest.write_bytes(content)
    output = Path("output.jsonl")

    res = CliRunner().invoke(app, [
        "infer", "--task", "asr", "--model", "openai/whisper-tiny", "--device", "cpu",
        "--input", str(manifest), "--output", str(output),
    ])
    assert res.exit_code == 1
    clean = " ".join(strip_ansi(res.output).split())
    assert "Cannot read manifest_mid_error.jsonl as UTF-8. Save the ASR manifest as UTF-8 and retry." in clean
    assert "UnicodeDecodeError" not in clean
    assert not output.exists()


def test_manifest_utf8_control_transcribes(monkeypatch):
    manifest = Path("valid_manifest.jsonl")
    manifest.write_text('{"audio": "clip.wav"}\n', encoding="utf-8")
    output = Path("output.jsonl")

    Path("clip.wav").touch()
    monkeypatch.setattr(infer, "_ASR_TRANSCRIBER_OVERRIDE", lambda path: "transcribed text")

    res = CliRunner().invoke(app, [
        "infer", "--task", "asr", "--model", "openai/whisper-tiny", "--device", "cpu",
        "--input", str(manifest), "--output", str(output),
    ])
    assert res.exit_code == 0
    assert output.exists()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["transcription"] == "transcribed text"


@pytest.mark.parametrize("special_name,expected_shown", [
    ("[red]manifest.jsonl", "[red]manifest.jsonl"),
    ("manifest[bold].jsonl", "manifest[bold].jsonl"),
    ("name\x1b[end].jsonl", "name[end].jsonl"),
])
def test_manifest_special_filename_printed_literally(monkeypatch, special_name, expected_shown):
    if os.name == "nt" and any(c in special_name for c in (":", "\\", "\x1b", "/", "*", "?", '"', "<", ">", "|")):
        pytest.skip("Invalid filename on Windows")

    manifest = Path(special_name)
    manifest.write_text('{"audio": "clip.wav"}\n', encoding="utf-16")
    output = Path("output.jsonl")

    res = CliRunner().invoke(app, [
        "infer", "--task", "asr", "--model", "openai/whisper-tiny", "--device", "cpu",
        "--input", str(manifest), "--output", str(output),
    ])
    assert res.exit_code == 1
    clean = " ".join(strip_ansi(res.output).split())
    assert f"Cannot read {expected_shown} as UTF-8. Save the ASR manifest as UTF-8 and retry." in clean
    assert "UnicodeDecodeError" not in clean
