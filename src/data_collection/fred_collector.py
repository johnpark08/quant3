from __future__ import annotations

from datetime import date
from pathlib import Path
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import numpy as np
import pandas as pd


def _fallback_series(name: str, index: int, start_date: str, end_date: str | None) -> pd.DataFrame:
    end = pd.Timestamp(end_date or date.today())
    dates = pd.date_range(start=start_date, end=end, freq="MS")
    rng = np.random.default_rng(7 + index)
    drift = 0.01 * (index + 1)
    shock = rng.normal(0, 0.4 + index * 0.1, size=len(dates))
    return pd.DataFrame({"date": dates, name: 100 + np.cumsum(drift + shock)})


def _read_fred_series(name: str, fred_code: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    params = {
        "id": fred_code,
        "cosd": start_date,
    }
    if end_date:
        params["coed"] = end_date

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; volatility-dashboard/1.0)",
        },
    )

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urlopen(request, timeout=20) as response:
                series = pd.read_csv(response)
            break
        except Exception as exc:
            last_error = exc
            if attempt < 1:
                sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"FRED download failed after retries: {last_error}")

    series = series.rename(columns={"observation_date": "date", fred_code: name})
    series["date"] = pd.to_datetime(series["date"])
    series[name] = pd.to_numeric(series[name].replace(".", np.nan), errors="coerce")
    return series[["date", name]]


def _read_yahoo_series(name: str, yahoo_ticker: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    import yfinance as yf

    downloaded = yf.download(
        yahoo_ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if downloaded.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {yahoo_ticker}.")

    value = downloaded["Close"]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]

    series = value.rename(name).reset_index()
    series.columns = ["date", name]
    series["date"] = pd.to_datetime(series["date"])
    series[name] = pd.to_numeric(series[name], errors="coerce")
    return series[["date", name]].dropna()


def _read_csv_fallback_series(
    name: str,
    url: str,
    value_column: str,
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    series = pd.read_csv(url)
    series = series.rename(columns={"date": "date", value_column: name})
    series["date"] = pd.to_datetime(series["date"])
    series[name] = pd.to_numeric(series[name], errors="coerce")
    series = series[["date", name]].dropna()
    series = series[series["date"] >= pd.Timestamp(start_date)]
    if end_date:
        series = series[series["date"] <= pd.Timestamp(end_date)]
    return series.reset_index(drop=True)


def _read_bls_ppi_series(name: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    start_year = str(pd.Timestamp(start_date).year)
    end_year = str(pd.Timestamp(end_date or date.today()).year)
    body = json.dumps({"seriesid": ["WPU00000000"], "startyear": start_year, "endyear": end_year}).encode("utf-8")
    request = Request(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API request failed: {payload.get('message')}")

    rows = []
    for item in payload["Results"]["series"][0]["data"]:
        period = item["period"]
        if not period.startswith("M"):
            continue
        rows.append(
            {
                "date": pd.Timestamp(year=int(item["year"]), month=int(period[1:]), day=1),
                name: float(item["value"]),
            }
        )

    series = pd.DataFrame(rows).sort_values("date")
    series = series[series["date"] >= pd.Timestamp(start_date)]
    if end_date:
        series = series[series["date"] <= pd.Timestamp(end_date)]
    return series.reset_index(drop=True)


def _read_federal_reserve_ip_series(name: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    request = Request(
        "https://www.federalreserve.gov/releases/g17/Current/ipdisk/ip_sa.txt",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 14 or parts[0].strip('"') != "B50001":
            continue
        year = int(parts[1])
        for month, value in enumerate(parts[2:14], start=1):
            if value.lower() in {"na", "."}:
                continue
            rows.append({"date": pd.Timestamp(year=year, month=month, day=1), name: float(value)})

    if not rows:
        raise RuntimeError("Federal Reserve G.17 total industrial production series B50001 was not found.")

    series = pd.DataFrame(rows).sort_values("date")
    series = series[series["date"] >= pd.Timestamp(start_date)]
    if end_date:
        series = series[series["date"] <= pd.Timestamp(end_date)]
    return series.reset_index(drop=True)


def _merge_series(frames: list[pd.DataFrame]) -> pd.DataFrame:
    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="date", how="outer")
    macro["date"] = pd.to_datetime(macro["date"])
    return macro.sort_values("date").reset_index(drop=True)


def collect_macro(indicators: dict[str, str], start_date: str, end_date: str | None = None) -> pd.DataFrame:
    cache_dir = Path("data/cache/fred")
    cache_dir.mkdir(parents=True, exist_ok=True)
    yahoo_fallbacks = {
        "wti_oil": "CL=F",
        "usd_krw": "KRW=X",
        "dollar_index": "DX-Y.NYB",
        "vix": "^VIX",
    }
    csv_fallbacks = {
        "ppi": ("https://eco3min.fr/dataset/us-ppi.csv", "ppi_index"),
        "industrial_production": ("https://eco3min.fr/dataset/us-industrial-production.csv", "indpro_index"),
    }
    official_fallbacks = {
        "ppi": (_read_bls_ppi_series, "bls"),
        "industrial_production": (_read_federal_reserve_ip_series, "federal_reserve"),
    }
    frames = []
    sources = []

    for index, (name, fred_code) in enumerate(indicators.items()):
        cache_path = cache_dir / f"{fred_code}.csv"
        try:
            if name in yahoo_fallbacks:
                series = _read_yahoo_series(name, yahoo_fallbacks[name], start_date, end_date)
                source = "yahoo"
            elif name in official_fallbacks:
                reader, source = official_fallbacks[name]
                series = reader(name, start_date, end_date)
            elif cache_path.exists():
                series = pd.read_csv(cache_path)
                series["date"] = pd.to_datetime(series["date"])
                source = "cache"
            else:
                series = _read_fred_series(name, fred_code, start_date, end_date)
                series.to_csv(cache_path, index=False)
                source = "fred"
        except Exception as exc:
            if name in yahoo_fallbacks:
                try:
                    series = _read_fred_series(name, fred_code, start_date, end_date)
                    series.to_csv(cache_path, index=False)
                    source = "fred"
                    print(f"[WARN] Yahoo Finance {yahoo_fallbacks[name]} failed; using FRED {fred_code}. Reason: {exc}")
                except Exception as yahoo_exc:
                    if cache_path.exists():
                        series = pd.read_csv(cache_path)
                        series["date"] = pd.to_datetime(series["date"])
                        source = "cache"
                        print(
                            f"[WARN] Yahoo Finance {yahoo_fallbacks[name]} and FRED {fred_code} failed; using cached data. "
                            f"Reasons: {exc}; {yahoo_exc}"
                        )
                    else:
                        series = _fallback_series(name, index, start_date, end_date)
                        source = "sample"
                        print(
                            f"[WARN] Yahoo Finance {yahoo_fallbacks[name]} and FRED {fred_code} failed; using generated sample data. "
                            f"Reasons: {exc}; {yahoo_exc}"
                        )
            elif name in official_fallbacks:
                try:
                    reader, source = official_fallbacks[name]
                    series = reader(name, start_date, end_date)
                    print(f"[WARN] FRED {fred_code} failed; using official fallback source. Reason: {exc}")
                except Exception as official_exc:
                    try:
                        url, value_column = csv_fallbacks[name]
                        series = _read_csv_fallback_series(name, url, value_column, start_date, end_date)
                        source = "fred_mirror"
                        print(
                            f"[WARN] FRED {fred_code} and official fallback failed; using FRED mirror CSV. "
                            f"Reasons: {exc}; {official_exc}"
                        )
                    except Exception as mirror_exc:
                        if cache_path.exists():
                            series = pd.read_csv(cache_path)
                            series["date"] = pd.to_datetime(series["date"])
                            source = "cache"
                            print(
                                f"[WARN] FRED {fred_code}, official fallback, and mirror failed; using cached data. "
                                f"Reasons: {exc}; {official_exc}; {mirror_exc}"
                            )
                        else:
                            series = _fallback_series(name, index, start_date, end_date)
                            source = "sample"
                            print(
                                f"[WARN] FRED {fred_code}, official fallback, and mirror failed; using generated sample data. "
                                f"Reasons: {exc}; {official_exc}; {mirror_exc}"
                            )
            elif cache_path.exists():
                series = pd.read_csv(cache_path)
                series["date"] = pd.to_datetime(series["date"])
                source = "cache"
                print(f"[WARN] FRED {fred_code} failed; using cached data. Reason: {exc}")
            else:
                series = _fallback_series(name, index, start_date, end_date)
                source = "sample"
                print(f"[WARN] FRED {fred_code} failed; using generated sample data. Reason: {exc}")

        if source in {"fred", "bls", "federal_reserve"}:
            series.to_csv(cache_path, index=False)

        frames.append(series)
        sources.append({"indicator": name, "fred_code": fred_code, "source": source, "rows": len(series)})

    macro = _merge_series(frames)
    macro.attrs["sources"] = pd.DataFrame(sources)
    return macro
