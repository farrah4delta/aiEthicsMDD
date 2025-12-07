"""Utility functions module."""
import json
from pathlib import Path
from typing import Any, Dict, Union


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save JSON data to file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_markdown(content: str, filepath: Union[str, Path]) -> None:
    """Save Markdown content to file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

