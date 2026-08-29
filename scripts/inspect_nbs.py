from __future__ import annotations

import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import openpyxl
import requests

RESOURCES = {
    "dec_2024": "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/download/1151",
    "jan_2025": "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/download/1230",
    "mar_2025": "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/download/1240",
    "may_2026": "https://microdata.nigerianstat.gov.ng/index.php/catalog/162/download/1427",
}


def workbook_payloads(content: bytes, content_type: str) -> list[tuple[str, bytes]]:
    if content[:2] == b"PK":
        try:
            with zipfile.ZipFile(BytesIO(content)) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xlsm"))]
                if names:
                    return [(Path(n).name, zf.read(n)) for n in names]
        except zipfile.BadZipFile:
            pass
    return [("download.xlsx", content)]


def inspect_book(name: str, payload: bytes) -> dict:
    wb = openpyxl.load_workbook(BytesIO(payload), read_only=True, data_only=True)
    result = {"name": name, "sheets": []}
    for ws in wb.worksheets:
        rows = []
        for idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            vals = [None if v is None else str(v) for v in row[:16]]
            if any(v not in (None, "") for v in vals):
                rows.append({"row": idx, "values": vals})
            if len(rows) >= 20:
                break
        result["sheets"].append({
            "title": ws.title,
            "max_row": ws.max_row,
            "max_column": ws.max_column,
            "sample": rows,
        })
    return result


def main(out_path: str) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "Nsat/0.3 (+https://github.com/FPC-effortless/Nsat)"})
    report = {}
    for key, url in RESOURCES.items():
        response = session.get(url, timeout=120)
        response.raise_for_status()
        payloads = workbook_payloads(response.content, response.headers.get("content-type", ""))
        report[key] = {
            "url": url,
            "status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "bytes": len(response.content),
            "workbooks": [inspect_book(name, data) for name, data in payloads],
        }
        print(json.dumps({key: report[key]}, indent=2))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/nbs/schema_report.json")
