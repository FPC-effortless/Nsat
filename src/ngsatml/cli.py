from __future__ import annotations

import argparse
import json
from pathlib import Path

from .boundaries import download_adm1, filter_states_geojson
from .config import load_config
from .dataset import build_market_satellite_dataset
from .geometry import state_bboxes
from .nbs import discover_downloads, save_resource_index
from .sources import SOURCES
from .stac import CatalogQuery, month_windows, search_items, write_manifest


def _require_states(cfg: dict, command: str) -> None:
    if not cfg.get("states"):
        raise SystemExit(f"{command} requires at least one state in the top-level states list")


def cmd_sources(_: argparse.Namespace) -> None:
    print(json.dumps(SOURCES, indent=2))


def cmd_boundaries(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    _require_states(cfg, "boundaries")
    raw = download_adm1(Path(args.output) / "boundaries")
    target = Path(args.output) / "boundaries" / "pilot_states.geojson"
    filter_states_geojson(raw, cfg["states"], target)
    print(target)


def cmd_catalog(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    _require_states(cfg, "catalog")
    boundary_path = Path(args.output) / "boundaries" / "pilot_states.geojson"
    if not boundary_path.exists():
        raise SystemExit("Run the boundaries command first")
    bboxes = state_bboxes(boundary_path)
    s2 = cfg.get("sentinel2", {})
    s1 = cfg.get("sentinel1", {})
    for state in cfg["states"]:
        bbox = bboxes[state]
        for start, end in month_windows(cfg["start_date"], cfg["end_date"]):
            for collection, cloud in [
                (s2.get("collection", "sentinel-2-l2a"), s2.get("cloud_cover_max", 60)),
                (s1.get("collection", "sentinel-1-grd"), None),
            ]:
                q = CatalogQuery(collection=collection, bbox=bbox, start=start, end=end, cloud_cover_max=cloud)
                items = search_items(q)
                slug = state.lower().replace(" ", "-")
                month = start[:7]
                path = Path(args.output) / "catalog" / collection / slug / f"{month}.json"
                write_manifest(items, path)
                print(f"{state} {month} {collection}: {len(items)}")


def cmd_nbs(args: argparse.Namespace) -> None:
    load_config(args.config)
    resources = discover_downloads()
    target = save_resource_index(resources, Path(args.output) / "nbs" / "resource_index.csv")
    print(f"Discovered {len(resources)} NBS download resources -> {target}")


def cmd_dataset(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    summary = build_market_satellite_dataset(
        cfg,
        args.output,
        market_month_limit=args.max_market_months,
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ngsatml")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sources")
    s.set_defaults(func=cmd_sources)
    for name, func in [("boundaries", cmd_boundaries), ("catalog", cmd_catalog), ("nbs", cmd_nbs)]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/pilot.yaml")
        sp.add_argument("--output", default="data")
        sp.set_defaults(func=func)

    ds = sub.add_parser("dataset")
    ds.add_argument("--config", default="configs/dataset-smoke.yaml")
    ds.add_argument("--output", default="data/dataset")
    ds.add_argument("--max-market-months", type=int, default=None)
    ds.set_defaults(func=cmd_dataset)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
