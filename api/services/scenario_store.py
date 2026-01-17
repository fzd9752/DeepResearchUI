import json
from functools import lru_cache
from pathlib import Path
from typing import List, Dict


@lru_cache
def load_scenarios() -> List[Dict]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "scenarios.json"
    if not data_path.exists():
        return []
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)
