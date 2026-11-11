from __future__ import annotations

import calendar
from datetime import date


def add_months(d: date, n: int) -> date:
    """Add n calendar months, clamping the day to the target month's length."""
    idx = d.year * 12 + (d.month - 1) + n
    y, m = divmod(idx, 12)
    m += 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def month_index(d: date, as_of: date) -> int:
    """Whole calendar months from as_of's month to d's month (0 = same month)."""
    return (d.year - as_of.year) * 12 + d.month - as_of.month


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"
