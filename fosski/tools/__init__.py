"""
FOSS-KI Tool Plugin System
============================
Each tool is a Python file with:
  - description: str — what the tool does (used for intent matching)
  - keywords: list[str] — trigger words
  - execute(args: dict) -> dict — the tool logic

Drop a new .py file in tools/ → system auto-discovers it.
Self-extension: Metacognition detects gaps → writes new tool files.
"""
