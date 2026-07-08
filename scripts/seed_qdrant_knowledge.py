from __future__ import annotations

import json
import os
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "knowledge_base" / "keyboard_marketing_knowledge_v1.jsonl"
SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"

DEFAULT_VECTOR_SIZES = {
    "marketing_cases": 384,
    "marketing_knowledge_base": 384,
}


def read_config() -> dict[str, str]:
    config: dict[str, str] = {}
    if SECRETS_PATH.exists():
        raw = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8-sig"))
        config.update({key: str(value) for key, value in raw.items() if value})
    for key in ("QDRANT_URL", "QDRANT_API_KEY"):
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    try:
        return bool(client.collection_exists(collection_name=collection_name))
    except Exception:
        try:
            client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False


def vector_size_from_info(info: Any, default: int) -> int:
    vectors = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(vectors, "vectors", None)
    if hasattr(vectors, "size"):
        return int(vectors.size)
    if isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        if hasattr(first, "size"):
            return int(first.size)
    return default


def ensure_collection(client: QdrantClient, collection_name: str) -> int:
    default_size = DEFAULT_VECTOR_SIZES.get(collection_name, 384)
    if collection_exists(client, collection_name):
        info = client.get_collection(collection_name=collection_name)
        return vector_size_from_info(info, default_size)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=default_size, distance=models.Distance.COSINE),
    )
    return default_size


def make_vector(size: int) -> list[float]:
    if size <= 0:
        return [1.0]
    vector = [0.0] * size
    vector[0] = 1.0
    return vector


def point_id(kb_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, kb_id))


def upsert_rows(client: QdrantClient, rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    by_collection: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_collection.setdefault(row["collection"], []).append(row)

    for collection_name, collection_rows in by_collection.items():
        vector_size = ensure_collection(client, collection_name)
        vector = make_vector(vector_size)
        points = []
        for row in collection_rows:
            payload = {key: value for key, value in row.items() if key != "collection"}
            payload["seed_source"] = DATA_PATH.name
            points.append(
                models.PointStruct(
                    id=point_id(str(row["kb_id"])),
                    vector=vector,
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection_name, points=points)
        counts[collection_name] = len(points)
    return counts


def main() -> int:
    config = read_config()
    url = config.get("QDRANT_URL", "")
    api_key = config.get("QDRANT_API_KEY", "")
    if not url or not api_key:
        print("Qdrant config missing. Set QDRANT_URL and QDRANT_API_KEY in Streamlit Secrets or environment.")
        return 1

    rows = load_rows(DATA_PATH)
    client = QdrantClient(url=url, api_key=api_key)
    counts = upsert_rows(client, rows)
    total = sum(counts.values())
    print(f"Seeded {total} knowledge records.")
    for collection_name, count in sorted(counts.items()):
        print(f"- {collection_name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
