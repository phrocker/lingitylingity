from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest


@pytest.fixture
def recommendation_fixture() -> dict[str, str]:
    path = Path(__file__).parent / "fixtures" / "recommended-decision.json"
    return cast(dict[str, str], json.loads(path.read_text(encoding="utf-8")))
