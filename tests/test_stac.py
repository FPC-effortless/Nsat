from ngsatml.stac import month_windows


def test_month_windows_half_open():
    assert list(month_windows("2025-11-01", "2026-02-01")) == [
        ("2025-11-01", "2025-12-01"),
        ("2025-12-01", "2026-01-01"),
        ("2026-01-01", "2026-02-01"),
    ]
