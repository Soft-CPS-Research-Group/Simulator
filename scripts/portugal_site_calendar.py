"""Explicit fictional site opening calendars, in Europe/Lisbon civil time."""

from __future__ import annotations
import numpy as np
import pandas as pd

HOLIDAYS = {
    "01-01",
    "04-07",
    "04-09",
    "04-25",
    "05-01",
    "06-08",
    "06-10",
    "08-15",
    "10-05",
    "11-01",
    "12-01",
    "12-08",
    "12-25",
    "06-13",
}


def service_hours(member, day):
    """Return [opening, closing) civil hours, or None for a closed date.

    Offices exclude weekends/holidays; retail, cafés, clinics and workshops
    have their own Saturday/Sunday rules. A care home remains staffed 24/7.
    """
    kind = member["routine_archetype"]
    i = member["member_index"]
    dow = day.dayofweek
    md = day.strftime("%m-%d")
    holiday = md in (HOLIDAYS if day.year == 2023 else (HOLIDAYS - {"04-07", "04-09", "06-08"}) | {"03-29", "03-31", "05-30"})
    vacation = (
        member["vacation_start_day"]
        <= day.dayofyear
        < member["vacation_start_day"] + member["vacation_days"]
    )
    if kind == "care_home":
        return (0.0, 24.0)
    if vacation:
        return None
    if kind == "office":
        return (8.0, 19.0) if dow < 5 and not holiday else None
    if kind == "clinic":
        return (
            None if holiday or dow == 6 else ((9.0, 13.0) if dow == 5 else (8.0, 20.0))
        )
    if kind == "workshop":
        return (
            None if holiday or dow == 6 else ((9.0, 13.0) if dow == 5 else (8.0, 18.0))
        )
    if md in ("01-01", "12-25"):
        return None
    if kind == "retail":
        if holiday or dow == 6:
            return (10.0, 17.0) if i % 2 == 0 else None
        return (9.5, 19.0) if dow == 5 else (9.0, 20.0)
    if kind == "cafe":
        return (8.0, 15.0) if dow == 6 or holiday else (7.0, 22.0)
    if kind == "grocery":
        return (9.0, 19.0) if dow == 6 or holiday else (8.0, 21.0)
    if kind == "sports_centre":
        return (8.0, 20.0) if dow >= 5 or holiday else (7.0, 23.0)
    raise ValueError(kind)


def opening_series(member, calendar):
    local = pd.DatetimeIndex(calendar.timestamp_utc).tz_convert("Europe/Lisbon")
    lookup = {d: service_hours(member, pd.Timestamp(d)) for d in set(local.date)}
    return np.array(
        [
            bool(lookup[d] is not None and lookup[d][0] <= h + mi / 60 < lookup[d][1])
            for d, h, mi in zip(local.date, local.hour, local.minute)
        ]
    )


def calendar_description(member):
    kind = member["routine_archetype"]
    return {
        "office": "2.ª-6.ª 08-19h; fecha fins de semana, feriados e férias",
        "retail": (
            "2.ª-6.ª 09-20h; sábado 09:30-19h; domingos/feriados 10-17h, exceto Ano Novo/Natal"
            if member["member_index"] % 2 == 0
            else "2.ª-6.ª 09-20h; sábado 09:30-19h; fecha domingos e feriados"
        ),
        "clinic": "2.ª-6.ª 08-20h; sábado 09-13h; fecha domingos e feriados",
        "workshop": "2.ª-6.ª 08-18h; sábado 09-13h; fecha domingos, feriados e férias",
        "cafe": "2.ª-sábado 07-22h; domingos/feriados 08-15h; fecha Ano Novo/Natal",
        "sports_centre": "dias úteis 07-23h; fim de semana/feriados 08-20h; fecha Ano Novo/Natal",
        "grocery": "2.ª-sábado 08-21h; domingos/feriados 09-19h; fecha Ano Novo/Natal",
        "care_home": "24 horas por dia, todos os dias, incluindo feriados",
    }[kind]
