from __future__ import annotations

import html as html_lib
import json
import re
import zipfile
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .cohd import add_cohd_targets, consolidate_cohd, parse_state_cohd_workbook
from .cpi import add_cpi_targets, consolidate_cpi, parse_state_food_cpi_workbook
from .nbs import NBSResource, NIGERIA_STATES, month_from_text
from .nbs_mirror import load_all_mirror_targets

NBS_CPI_CATALOG = "https://microdata.nigerianstat.gov.ng/index.php/catalog/154/related-materials"
NBS_COHD_CATALOG = "https://microdata.nigerianstat.gov.ng/index.php/catalog/146/related-materials"


def _session(*, retries: int = 1) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    s = requests.Session()
    s.headers.update({"User-Agent": "Nsat/0.4 (+https://github.com/FPC-effortless/Nsat)"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _clean_html(value: str) -> str:
    text = html_lib.unescape(value)
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def discover_catalog_resources(
    catalog_url: str,
    *,
    source: str,
    session: requests.Session | None = None,
) -> list[NBSResource]:
    """Discover NBS resources directly when the NADA host is available."""
    s = session or _session()
    response = s.get(catalog_url, timeout=(8, 15))
    response.raise_for_status()
    body = response.text
    catalog_match = re.search(r"/catalog/(\d+)", catalog_url)
    if not catalog_match:
        raise ValueError(f"Could not infer NBS catalog id from {catalog_url}")
    catalog_id = catalog_match.group(1)
    pattern = re.compile(rf'href=["\']([^"\']*/catalog/{re.escape(catalog_id)}/download/\d+)["\']', re.I)
    found: dict[str, NBSResource] = {}
    for match in pattern.finditer(body):
        url = urljoin(catalog_url, match.group(1))
        context = _clean_html(body[max(0, match.start() - 1600):match.end() + 300])
        found.setdefault(
            url,
            NBSResource(
                title=context[-700:] if context else f"NBS catalog {catalog_id} resource",
                url=url,
                month=month_from_text(context),
                source_page=catalog_url,
                source=source,
            ),
        )
    if not found:
        raise RuntimeError(f"No downloadable resources discovered at {catalog_url}")
    return list(found.values())


def _workbook_payloads(content: bytes, fallback_name: str) -> list[tuple[str, bytes]]:
    if zipfile.is_zipfile(BytesIO(content)):
        with zipfile.ZipFile(BytesIO(content)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/") and n.lower().endswith((".xlsx", ".xlsm", ".xls"))]
            return [(Path(name).name, zf.read(name)) for name in names]
    if content.startswith(b"PK") or content.startswith(b"\xd0\xcf\x11\xe0"):
        return [(fallback_name, content)]
    return []


def _download_workbooks(resource: NBSResource, session: requests.Session) -> list[tuple[str, bytes]]:
    response = session.get(resource.url, timeout=(8, 30))
    response.raise_for_status()
    fallback = Path(resource.url.split("?", 1)[0]).name or "download.xlsx"
    return _workbook_payloads(response.content, fallback)


def _regime_for_report_month(month: pd.Timestamp | None) -> str:
    return "2024-base" if month is not None and month >= pd.Timestamp("2025-01-01") else "2009-11-base"


def _filter_dates(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[(frame["month"] >= start) & (frame["month"] < end)].copy()


def audit_panel(frame: pd.DataFrame, *, value_col: str, regime_col: str | None = None) -> dict[str, object]:
    if frame.empty:
        return {"ok": False, "rows": 0, "reason": "empty", "incomplete_months": [], "missing_months": [], "unknown_states": []}
    groups = [regime_col, "month"] if regime_col else ["month"]
    groups = [g for g in groups if g is not None]
    counts = frame.groupby(groups)["state"].nunique()
    incomplete: list[dict[str, object]] = []
    for key, count in counts.items():
        if int(count) != len(NIGERIA_STATES):
            key_tuple = key if isinstance(key, tuple) else (key,)
            incomplete.append({"group": [str(x) for x in key_tuple], "state_count": int(count)})

    missing_months: list[dict[str, object]] = []
    regime_values: Iterable[tuple[str | None, pd.DataFrame]]
    regime_values = frame.groupby(regime_col, dropna=False) if regime_col else [(None, frame)]
    for regime, part in regime_values:
        months = sorted(pd.Timestamp(x) for x in part["month"].dropna().unique())
        if not months:
            continue
        observed = set(months)
        missing = [m.strftime("%Y-%m") for m in pd.date_range(months[0], months[-1], freq="MS") if m not in observed]
        if missing:
            missing_months.append({"regime": None if regime is None else str(regime), "months": missing})

    unknown_states = sorted(set(frame["state"].dropna()) - set(NIGERIA_STATES))
    duplicate_keys = int(frame.duplicated(groups + ["state"]).sum())
    nonpositive = int((pd.to_numeric(frame[value_col], errors="coerce") <= 0).fillna(False).sum())
    null_values = int(frame[value_col].isna().sum())
    ok = not incomplete and not missing_months and not unknown_states and duplicate_keys == 0 and nonpositive == 0 and null_values == 0
    return {
        "ok": ok,
        "rows": int(len(frame)),
        "first_month": pd.Timestamp(frame["month"].min()).strftime("%Y-%m"),
        "last_month": pd.Timestamp(frame["month"].max()).strftime("%Y-%m"),
        "months": int(frame["month"].nunique()),
        "states": int(frame["state"].nunique()),
        "incomplete_months": incomplete,
        "missing_months": missing_months,
        "unknown_states": unknown_states,
        "duplicate_keys": duplicate_keys,
        "nonpositive_values": nonpositive,
        "null_values": null_values,
    }


def _native_frames(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    session: requests.Session,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict[str, object]], list[dict[str, str]]]:
    cpi_frames: list[pd.DataFrame] = []
    cohd_frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for kind, url in (("cpi", NBS_CPI_CATALOG), ("cohd", NBS_COHD_CATALOG)):
        try:
            resources = discover_catalog_resources(url, source=f"nada-{kind}", session=session)
        except Exception as exc:
            errors.append({"kind": kind, "url": url, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for resource in resources:
            sources.append({"kind": kind, **asdict(resource), "transport": "nbs-native"})
            try:
                books = _download_workbooks(resource, session)
            except Exception as exc:
                errors.append({"kind": kind, "url": resource.url, "error": f"{type(exc).__name__}: {exc}"})
                continue
            for name, payload in books:
                report_month = month_from_text(name) or resource.month
                try:
                    if kind == "cpi":
                        frame = parse_state_food_cpi_workbook(
                            payload,
                            source_url=resource.url,
                            workbook_name=name,
                            index_regime=_regime_for_report_month(report_month),
                        )
                        if not frame.empty:
                            cpi_frames.append(_filter_dates(frame, start, end))
                    elif report_month is not None and start <= report_month < end:
                        frame = parse_state_cohd_workbook(
                            payload,
                            month=report_month,
                            source_url=resource.url,
                            workbook_name=name,
                        )
                        if not frame.empty:
                            cohd_frames.append(frame)
                except Exception as exc:
                    errors.append({"kind": kind, "url": resource.url, "error": f"{name}: {type(exc).__name__}: {exc}"})
    return cpi_frames, cohd_frames, sources, errors


def _source_index_from_frames(cpi_frames: list[pd.DataFrame], cohd_frames: list[pd.DataFrame], repos: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for kind, frames in (("cpi", cpi_frames), ("cohd", cohd_frames)):
        for frame in frames:
            cols = [c for c in ("source_resource_id", "source_resource", "source_url", "mirror_repo", "transport") if c in frame.columns]
            if not cols:
                continue
            for row in frame[cols].drop_duplicates().to_dict(orient="records"):
                records.append({"kind": kind, **row})
    for repo in repos:
        records.append({"kind": "mirror-discovery", "mirror_repo": repo, "transport": "huggingface-mirror"})
    return pd.DataFrame(records).drop_duplicates() if records else pd.DataFrame()


def build_nbs_state_targets(
    output_dir: str | Path,
    *,
    start_date: str | pd.Timestamp = "2024-10-01",
    end_date: str | pd.Timestamp | None = None,
    strict: bool = True,
    native_refresh: bool = False,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Build all-state NBS Food CPI and CoHD target layers.

    Canonical publisher is always Nigeria NBS. By default, materialization uses
    Electric Sheep Africa's provenance-preserving Parquet transport mirror so
    GitHub builds do not depend on the intermittently unreachable NBS NADA host.
    `native_refresh=True` additionally attempts direct NBS downloads and
    consolidates them with the same quality gates.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date).to_period("M").to_timestamp()
    if end_date is None:
        now = pd.Timestamp.utcnow().tz_localize(None)
        end = now.to_period("M").to_timestamp() + pd.DateOffset(months=1)
    else:
        end = pd.Timestamp(end_date).to_period("M").to_timestamp()
    s = session or _session()

    mirror_cpi, mirror_cohd, mirror_errors, repos = load_all_mirror_targets(session=s)
    cpi_frames = [_filter_dates(f, start, end) for f in mirror_cpi if not f.empty]
    cohd_frames = [_filter_dates(f, start, end) for f in mirror_cohd if not f.empty]
    transport_errors: list[dict[str, str]] = list(mirror_errors)
    native_sources: list[dict[str, object]] = []
    if native_refresh:
        ncpi, ncohd, native_sources, native_errors = _native_frames(start=start, end=end, session=s)
        cpi_frames.extend(ncpi)
        cohd_frames.extend(ncohd)
        transport_errors.extend(native_errors)

    cpi = add_cpi_targets(_filter_dates(consolidate_cpi(cpi_frames), start, end))
    cohd = add_cohd_targets(_filter_dates(consolidate_cohd(cohd_frames), start, end))
    cpi_audit = audit_panel(cpi, value_col="food_cpi", regime_col="index_regime")
    cohd_audit = audit_panel(cohd, value_col="cohd_ngn_person_day")

    if not cpi.empty:
        preferred = cpi.assign(_priority=cpi["index_regime"].eq("2024-base").astype(int))
        preferred = preferred.sort_values(["state", "month", "_priority"]).drop_duplicates(["state", "month"], keep="last").drop(columns="_priority")
    else:
        preferred = cpi.copy()
    state_month = preferred.merge(cohd, on=["state", "month"], how="outer", suffixes=("_cpi", "_cohd"))
    if not state_month.empty:
        state_month = state_month.sort_values(["month", "state"]).reset_index(drop=True)

    source_index = _source_index_from_frames(cpi_frames, cohd_frames, repos)
    if native_sources:
        source_index = pd.concat([source_index, pd.DataFrame(native_sources)], ignore_index=True, sort=False)
    source_index.to_csv(output / "nbs_source_index.csv", index=False)
    cpi.to_parquet(output / "nbs_food_cpi.parquet", index=False)
    cpi.to_csv(output / "nbs_food_cpi.csv", index=False)
    cohd.to_parquet(output / "nbs_cohd.parquet", index=False)
    cohd.to_csv(output / "nbs_cohd.csv", index=False)
    state_month.to_parquet(output / "nbs_state_month_targets.parquet", index=False)

    quality = {
        "canonical_publisher": "National Bureau of Statistics, Nigeria",
        "transport_policy": "Hugging Face mirror is transport only; original NBS resource IDs/URLs are retained",
        "strict": bool(strict),
        "native_refresh": bool(native_refresh),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date_exclusive": end.strftime("%Y-%m-%d"),
        "cpi": cpi_audit,
        "cohd": cohd_audit,
        "transport_errors": transport_errors,
        "mirror_repo_count": len(repos),
    }
    (output / "nbs_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    summary = {
        "version": "2.0.0",
        "canonical_publisher": "National Bureau of Statistics, Nigeria",
        "transport": "huggingface-mirror" + ("+nbs-native" if native_refresh else ""),
        "mirror_repo_count": len(repos),
        "cpi_rows": int(len(cpi)),
        "cpi_months": int(cpi["month"].nunique()) if not cpi.empty else 0,
        "cpi_latest": cpi["month"].max().strftime("%Y-%m") if not cpi.empty else None,
        "cpi_regimes": sorted(cpi["index_regime"].dropna().unique().tolist()) if not cpi.empty else [],
        "cohd_rows": int(len(cohd)),
        "cohd_months": int(cohd["month"].nunique()) if not cohd.empty else 0,
        "cohd_latest": cohd["month"].max().strftime("%Y-%m") if not cohd.empty else None,
        "state_month_rows": int(len(state_month)),
        "cpi_quality_ok": bool(cpi_audit["ok"]),
        "cohd_quality_ok": bool(cohd_audit["ok"]),
        "transport_error_count": len(transport_errors),
        "output_dir": str(output),
    }
    (output / "nbs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if strict and (not cpi_audit["ok"] or not cohd_audit["ok"]):
        raise RuntimeError(f"NBS target quality gate failed: {json.dumps(quality, indent=2)}")
    return summary
