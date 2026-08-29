from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

CPI_REPOS = [
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-04171bb4",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-16b00a9b",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-1b50c949",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-1c054a4a",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-3f8317b3",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-4a0e9421",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-51535e66",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-6196e568",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-7d6a1c71",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-87c6870e",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-99cb3ab8",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-af2ed6cd",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-b4303a07",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-caed8cff",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-d457a2ce",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-d60bb295",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-d7f5055d",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-dc0e7a8c",
    "electricsheepafrica/africa-nigeria-consumer-price-index-and-inflation-f16262fc",
]


def discover_repos(search: str) -> list[str]:
    r = requests.get(
        "https://huggingface.co/api/datasets",
        params={"author": "electricsheepafrica", "search": search, "limit": 100, "full": "true"},
        timeout=30,
    )
    r.raise_for_status()
    return sorted({row["id"] for row in r.json() if row.get("id")})


def load_df(repo: str) -> tuple[pd.DataFrame, str]:
    r = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": repo},
        timeout=30,
    )
    r.raise_for_status()
    files = r.json().get("parquet_files", [])
    if not files:
        raise RuntimeError("no parquet files")
    parts = []
    for item in files:
        p = requests.get(item["url"], timeout=60)
        p.raise_for_status()
        parts.append(pd.read_parquet(BytesIO(p.content)))
    return pd.concat(parts, ignore_index=True), files[0]["url"]


def load_repo(repo: str) -> dict:
    df, url = load_df(repo)
    detail = {"repo": repo, "rows": len(df), "columns": list(df.columns), "parquet_url": url}
    for col in ["source_dataset", "source_resource", "source_resource_id", "source_url", "source_sheet", "retrieved_at", "state", "dimension_state", "indicator_name"]:
        if col in df.columns:
            detail[f"{col}_sample"] = df[col].dropna().astype(str).unique().tolist()[:80]
    detail["head"] = df.head(4).where(pd.notna(df.head(4)), None).to_dict(orient="records")
    if "state" in df.columns:
        state_rows = df[df["state"].notna()].head(4)
        detail["state_rows"] = [
            {k: (None if pd.isna(v) else v) for k, v in row.items() if not pd.isna(v)}
            for row in state_rows.to_dict(orient="records")
        ]
    return detail


def main(out_path: str) -> None:
    discovered = {
        "cpi": discover_repos("africa-nigeria-consumer-price-index-and-inflation"),
        "cohd": discover_repos("africa-nigeria-cost-of-healthy-diet"),
    }
    repos = sorted(set(CPI_REPOS + discovered["cpi"] + discovered["cohd"]))
    report = {"_discovered": discovered}
    for repo in repos:
        try:
            report[repo] = load_repo(repo)
        except Exception as exc:
            report[repo] = {"repo": repo, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(report[repo], indent=2, default=str))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not any(isinstance(v, dict) and "rows" in v for k, v in report.items() if not k.startswith("_")):
        raise SystemExit("No Hugging Face NBS mirrors were readable")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/nbs/hf_mirror_probe.json")
