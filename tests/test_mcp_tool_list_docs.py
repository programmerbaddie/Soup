"""Docs check for MCP default tool discovery and call-time restrictions (#3196)."""

from __future__ import annotations

from pathlib import Path


def _mcp_section_normalized() -> str:
    path = Path("docs/commands.md")
    assert path.is_file(), "docs/commands.md must exist"
    text = path.read_text(encoding="utf-8")
    heading = "## Fine-tune from your coding agent (MCP)"
    assert heading in text, f"'{heading}' heading must exist in docs/commands.md"
    start = text.index(heading)
    next_heading = "## Gate and Verdict Exit Codes"
    end = text.index(next_heading, start) if next_heading in text else len(text)
    return " ".join(text[start:end].split())


def test_mcp_tools_list_discovery_documented():
    content = _mcp_section_normalized()
    assert "`tools/list`" in content
    assert "18 tools" in content
    assert "14 are callable" in content
    assert "default flags" in content
