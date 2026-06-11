import json
import logging
from pathlib import Path

_idf_cache: dict[str, float] | None = None
_IDF_PATH = Path(__file__).parent.parent.parent / "Data" / "idf_weights.json"


def load_idf() -> dict[str, float]:
    global _idf_cache
    if _idf_cache is not None:
        return _idf_cache
    if not _IDF_PATH.exists():
        logging.warning(
            "IDF weights not found at %s. BM25 will use uniform IDF=1.0. "
            "Run Scripts/build_idf.py after downloading the LinkedIn Job Postings corpus.",
            _IDF_PATH,
        )
        _idf_cache = {}
        return _idf_cache
    with open(_IDF_PATH, encoding="utf-8") as f:
        _idf_cache = json.load(f)
    logging.info("Loaded IDF weights: %d terms", len(_idf_cache))
    return _idf_cache
