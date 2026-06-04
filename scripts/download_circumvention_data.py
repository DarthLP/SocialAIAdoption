"""
Script summary:
Download circumvention / adaptation proxies (Tor Metrics daily users + Google Trends VPN
and ChatGPT topic interest) for Italy's 2023 ChatGPT ban window, mirroring Kreitmeir & Raschky (2023).

Functionality:
- Tor Metrics: per-country relay and bridge user CSVs (verbatim HTTP bodies) for IT + controls.
- Google Trends: daily interest for topics "Virtual private network" and "ChatGPT" (not bare keywords).
- Combined tidy CSVs, provenance manifest, and data README under data/raw/circumvention/.
- Idempotent, per-source failure isolation, summary table on exit.

How to apply/run:
  .venv/bin/python scripts/download_circumvention_data.py

Manual Google Trends fallback (if rate-limited): export daily CSV from trends.google.com
into data/raw/circumvention/google_trends/gtrends_vpn_{GEO}_{START}_{END}.csv (or gtrends_chatgpt_*)
and re-run.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pytrends.request import TrendReq

# --- Study window and countries ---
START = "2023-01-01"
END = "2023-06-30"
BAN_DATE = "2023-03-31"
LIFT_DATE = "2023-04-28"
TREATED = "IT"
CONTROLS = ["DE", "FR", "ES", "GB", "US"]
COUNTRIES = [TREATED] + CONTROLS

# --- Google Trends VPN (paper: topic "Virtual Private Networks", not keyword VPN alone) ---
GOOGLE_TRENDS_TOPIC_LABEL = "Virtual private network"
GOOGLE_TRENDS_TOPIC_MID: str | None = None  # override e.g. "/m/0..." if suggestions fail

# --- Google Trends ChatGPT (attention/salience proxy, not usage) ---
GOOGLE_TRENDS_CHATGPT_TOPIC_LABEL = "ChatGPT"
GOOGLE_TRENDS_CHATGPT_TOPIC_MID: str | None = None

# Per-geo interest_over_time is rescaled 0-100 within that country and window only;
# cross-country levels are NOT comparable — within-country over-time movement only.

# --- Tor / HTTP ---
TOR_RELAY_URL = (
    "https://metrics.torproject.org/userstats-relay-country.csv"
    "?start={start}&end={end}&country={cc}"
)
TOR_BRIDGE_URL = (
    "https://metrics.torproject.org/userstats-bridge-country.csv"
    "?start={start}&end={end}&country={cc}"
)
TOR_USER_AGENT = (
    "SocialAIAdoption-circumvention-download/1.0 "
    "(academic research; Italy ChatGPT ban replication)"
)
TOR_DELAY_S = 2.5
TOR_TIMEOUT_S = 60
TOR_MAX_RETRIES = 3

TRENDS_DELAY_S = 7.0
TRENDS_MAX_RETRIES = 3
TRENDS_HL = "en-US"
TRENDS_TZ = 0

def _read_tor_csv(path: Path) -> pd.DataFrame:
    """Function summary: read Tor Metrics CSV skipping leading # comment lines."""
    return pd.read_csv(path, comment="#")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)


def _setup_project_root(caller_file: Path) -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py.

    Parameters:
    - caller_file: path to this script (__file__).

    Returns:
    - Absolute repository root Path.
    """
    for parent in caller_file.resolve().parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller_file)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root(Path(__file__))
OUT_DIR = PROJECT_ROOT / "data/raw/circumvention"
TOR_DIR = OUT_DIR / "tor"
GT_DIR = OUT_DIR / "google_trends"


def _git_commit_hash() -> str | None:
    """Function summary: return current git HEAD hash or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _fetch_url_verbatim(
    url: str,
    dest: Path,
    *,
    delay_after_s: float = TOR_DELAY_S,
) -> dict[str, Any]:
    """Function summary: GET url with retries and save response bytes verbatim.

    Parameters:
    - url: full request URL (logged).
    - dest: output file path.
    - delay_after_s: polite sleep after attempt completes.

    Returns:
    - Dict with status_code, row_count (if parseable), error, outfile.
    """
    record: dict[str, Any] = {
        "url": url,
        "outfile": str(dest.relative_to(PROJECT_ROOT)),
        "status_code": None,
        "row_count": None,
        "date_min": None,
        "date_max": None,
        "error": None,
    }
    headers = {"User-Agent": TOR_USER_AGENT}
    last_err: str | None = None

    for attempt in range(1, TOR_MAX_RETRIES + 1):
        try:
            LOG.info("GET %s (attempt %d/%d)", url, attempt, TOR_MAX_RETRIES)
            resp = requests.get(url, headers=headers, timeout=TOR_TIMEOUT_S)
            record["status_code"] = resp.status_code
            LOG.info("  status=%s bytes=%s", resp.status_code, len(resp.content))

            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}"
                if attempt < TOR_MAX_RETRIES:
                    time.sleep(2**attempt)
                continue

            if not resp.content or not resp.content.strip():
                last_err = "empty response body"
                if attempt < TOR_MAX_RETRIES:
                    time.sleep(2**attempt)
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)

            try:
                df = _read_tor_csv(dest)
                record["row_count"] = len(df)
                if "date" in df.columns and len(df) > 0:
                    dates = pd.to_datetime(df["date"], errors="coerce")
                    record["date_min"] = str(dates.min().date())
                    record["date_max"] = str(dates.max().date())
            except Exception as parse_exc:  # noqa: BLE001
                record["error"] = f"saved but parse failed: {parse_exc}"

            time.sleep(delay_after_s)
            return record

        except requests.RequestException as exc:
            last_err = str(exc)
            LOG.warning("  request error: %s", exc)
            if attempt < TOR_MAX_RETRIES:
                time.sleep(2**attempt)

    record["error"] = last_err or "unknown failure"
    time.sleep(delay_after_s)
    return record


def _tor_raw_path(kind: str, cc: str) -> Path:
    """Function summary: path for verbatim Tor raw CSV (relay or bridge)."""
    return TOR_DIR / f"userstats-{kind}-country_{cc}_{START}_{END}.csv"


def download_tor_all(manifest_sources: list[dict[str, Any]], failures: list[str]) -> None:
    """Function summary: download relay and bridge Tor CSVs for all countries."""
    for country in COUNTRIES:
        cc = country.lower()
        for kind, url_tpl in (
            ("relay", TOR_RELAY_URL),
            ("bridge", TOR_BRIDGE_URL),
        ):
            url = url_tpl.format(start=START, end=END, cc=cc)
            dest = _tor_raw_path(kind, cc)
            try:
                rec = _fetch_url_verbatim(url, dest)
                rec["name"] = f"tor_{kind}"
                rec["country"] = cc
                rec["params"] = {"start": START, "end": END, "country": cc}
                manifest_sources.append(rec)
                if rec.get("error") or rec.get("status_code") != 200:
                    failures.append(f"tor_{kind}:{cc}")
                elif rec.get("row_count") == 0:
                    LOG.warning(
                        "WARNING tor %s %s: empty rows for window %s–%s",
                        kind,
                        cc,
                        START,
                        END,
                    )
            except Exception as exc:  # noqa: BLE001
                LOG.exception("tor %s %s failed: %s", kind, cc, exc)
                failures.append(f"tor_{kind}:{cc}")
                manifest_sources.append(
                    {
                        "name": f"tor_{kind}",
                        "country": cc,
                        "url": url,
                        "error": str(exc),
                        "status_code": None,
                    }
                )


def _parse_cc_from_tor_filename(path: Path, kind: str) -> str | None:
    """Function summary: extract lowercase country code from Tor raw filename."""
    pat = rf"userstats-{kind}-country_([a-z]{{2}})_{re.escape(START)}_{re.escape(END)}\.csv"
    m = re.match(pat, path.name)
    return m.group(1) if m else None


def _combine_tor(kind: str, combined_name: str) -> pd.DataFrame | None:
    """Function summary: concatenate per-country Tor raw files into one tidy CSV.

    Parameters:
    - kind: 'relay' or 'bridge'.
    - combined_name: output filename under OUT_DIR.

    Returns:
    - Combined DataFrame or None if no files loaded.
    """
    pattern = f"userstats-{kind}-country_*_{START}_{END}.csv"
    paths = sorted(TOR_DIR.glob(pattern)) if TOR_DIR.is_dir() else []
    frames: list[pd.DataFrame] = []
    columns_logged = False

    for path in paths:
        cc = _parse_cc_from_tor_filename(path, kind)
        if cc is None:
            LOG.warning("skip unrecognized tor file: %s", path.name)
            continue
        try:
            df = _read_tor_csv(path)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("could not read %s: %s", path, exc)
            continue

        if not columns_logged:
            LOG.info("Tor %s columns (first file): %s", kind, df.columns.tolist())
            columns_logged = True

        if "date" not in df.columns or "users" not in df.columns:
            LOG.warning("WARNING %s missing date/users columns: %s", path.name, df.columns.tolist())
            continue

        df = df.copy()
        df["query_country"] = cc
        frames.append(df)

        dates = pd.to_datetime(df["date"], errors="coerce")
        if len(df) == 0:
            LOG.warning("WARNING tor %s %s: empty file", kind, cc)
        else:
            dmin, dmax = dates.min(), dates.max()
            if pd.isna(dmin) or dmin.date() > pd.Timestamp(START).date():
                LOG.warning(
                    "WARNING tor %s %s: starts after %s (min=%s)",
                    kind,
                    cc,
                    START,
                    dmin,
                )
            if pd.isna(dmax) or dmax.date() < pd.Timestamp(END).date():
                LOG.warning(
                    "WARNING tor %s %s: ends before %s (max=%s)",
                    kind,
                    cc,
                    END,
                    dmax,
                )

    if not frames:
        LOG.warning("No tor %s files combined", kind)
        return None

    combined = pd.concat(frames, ignore_index=True)
    out_path = OUT_DIR / combined_name
    combined.to_csv(out_path, index=False)
    LOG.info("Wrote %s (%d rows)", out_path, len(combined))
    return combined


def _resolve_topic_mid(
    pytrends: TrendReq,
    topic_label: str,
    topic_mid_override: str | None = None,
) -> tuple[str, str, str]:
    """Function summary: resolve Google Trends topic mid for a topic label.

    Parameters:
    - pytrends: initialized TrendReq client.
    - topic_label: Trends topic title to resolve.
    - topic_mid_override: optional fixed mid (skips suggestions).

    Returns:
    - Tuple (mid, title, query_type) with mid like '/m/...'.

    Raises:
    - RuntimeError: if override unset and no unambiguous topic match.
    """
    if topic_mid_override:
        return topic_mid_override, topic_label, "topic"

    suggestions = pytrends.suggestions(topic_label)
    if not suggestions:
        raise RuntimeError(
            f"No Google Trends suggestions for {topic_label!r}. "
            "Set topic_mid_override manually in the script."
        )

    label_norm = topic_label.strip().lower()
    topic_entries = [
        s for s in suggestions if str(s.get("type", "")).lower() == "topic"
    ]
    exact = [
        s
        for s in topic_entries
        if str(s.get("title", "")).strip().lower() == label_norm
    ]
    candidates = exact if exact else topic_entries
    if not candidates:
        candidates = suggestions
    if len(candidates) != 1:
        titles = [(c.get("title"), c.get("mid"), c.get("type")) for c in suggestions[:8]]
        raise RuntimeError(
            f"Ambiguous topic resolution for {topic_label!r}: "
            f"{len(candidates)} matches. Suggestions sample: {titles}. "
            "Set topic_mid_override in the script."
        )

    chosen = candidates[0]
    mid = str(chosen.get("mid", "")).strip()
    if not mid.startswith("/m/"):
        mid = f"/m/{mid.lstrip('/')}" if mid else ""
    if not mid:
        raise RuntimeError(f"Topic candidate has no mid: {chosen}")
    return mid, str(chosen.get("title", topic_label)), "topic"


def download_google_trends(
    pytrends: TrendReq,
    topic_mid: str,
    topic_title: str,
    manifest_sources: list[dict[str, Any]],
    failures: list[str],
    *,
    series: str = "vpn",
) -> pd.DataFrame | None:
    """Function summary: download daily Google Trends topic interest per country geo.

    Parameters:
    - pytrends: TrendReq client.
    - topic_mid: resolved topic mid.
    - topic_title: resolved topic title.
    - manifest_sources: provenance list to append.
    - failures: failure id list to append.
    - series: ``vpn`` or ``chatgpt`` (file prefix and interest column name).

    Returns:
    - Combined DataFrame or None if all geos failed.
    """
    if series == "chatgpt":
        file_prefix = "gtrends_chatgpt"
        out_col = "chatgpt_interest"
        manifest_name = "google_trends_chatgpt_topic"
        combined_name = "google_trends_chatgpt_by_country.csv"
    else:
        file_prefix = "gtrends_vpn"
        out_col = "vpn_interest"
        manifest_name = "google_trends_vpn_topic"
        combined_name = "google_trends_vpn_by_country.csv"

    timeframe = f"{START} {END}"
    frames: list[pd.DataFrame] = []

    for geo in COUNTRIES:
        dest = GT_DIR / f"{file_prefix}_{geo}_{START}_{END}.csv"
        record: dict[str, Any] = {
            "name": manifest_name,
            "country": geo,
            "url": "pytrends.interest_over_time",
            "params": {
                "topic_mid": topic_mid,
                "topic_title": topic_title,
                "timeframe": timeframe,
                "geo": geo,
                "hl": TRENDS_HL,
                "tz": TRENDS_TZ,
            },
            "outfile": str(dest.relative_to(PROJECT_ROOT)),
            "status_code": None,
            "row_count": None,
            "date_min": None,
            "date_max": None,
            "error": None,
        }

        success = False
        last_err: str | None = None
        query_modes: list[tuple[str, list[str], str]] = [
            ("topic", [topic_mid], topic_mid),
        ]
        if series == "chatgpt":
            query_modes.append(("keyword", [GOOGLE_TRENDS_CHATGPT_TOPIC_LABEL], ""))

        for qtype, kw_list, q_mid in query_modes:
            if success:
                break
            for attempt in range(1, TRENDS_MAX_RETRIES + 1):
                try:
                    LOG.info(
                        "Google Trends geo=%s %s=%s (attempt %d/%d)",
                        geo,
                        qtype,
                        kw_list[0],
                        attempt,
                        TRENDS_MAX_RETRIES,
                    )
                    pytrends.build_payload(
                        kw_list=kw_list,
                        timeframe=timeframe,
                        geo=geo,
                        gprop="",
                    )
                    df = pytrends.interest_over_time()
                    if df is None or df.empty:
                        last_err = "empty interest_over_time"
                        time.sleep(2**attempt + TRENDS_DELAY_S)
                        continue

                    df = df.reset_index()
                    if "isPartial" in df.columns:
                        df = df.drop(columns=["isPartial"])

                    api_col = None
                    for col in df.columns:
                        if col == "date":
                            continue
                        api_col = col
                        break
                    if api_col is None:
                        last_err = f"no interest column in {df.columns.tolist()}"
                        time.sleep(2**attempt + TRENDS_DELAY_S)
                        continue

                    tidy = df.rename(columns={"date": "date", api_col: out_col})
                    tidy = tidy[["date", out_col]].copy()
                    tidy["geo"] = geo
                    tidy["trends_query_type"] = qtype
                    tidy["trends_mid"] = q_mid

                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tidy.to_csv(dest, index=False)

                    record["status_code"] = 200
                    record["row_count"] = len(tidy)
                    dates = pd.to_datetime(tidy["date"], errors="coerce")
                    record["date_min"] = str(dates.min().date())
                    record["date_max"] = str(dates.max().date())
                    manifest_sources.append(record)
                    frames.append(tidy)
                    success = True
                    LOG.info(
                        "  rows=%d range=%s–%s (%s)",
                        len(tidy),
                        record["date_min"],
                        record["date_max"],
                        qtype,
                    )
                    break

                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                    err_lower = last_err.lower()
                    if "429" in err_lower or "rate" in err_lower:
                        LOG.warning(
                            "WARNING Google Trends rate-limited for %s. "
                            "Export manually from https://trends.google.com into %s and re-run.",
                            geo,
                            dest,
                        )
                    LOG.warning("  trends error: %s", exc)
                    time.sleep(2**attempt + TRENDS_DELAY_S)

        if not success:
            record["error"] = last_err
            manifest_sources.append(record)
            failures.append(f"google_trends_{series}:{geo}")
            LOG.warning(
                "WARNING google_trends %s failed after retries. Manual CSV path: %s",
                geo,
                dest,
            )

        time.sleep(TRENDS_DELAY_S)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    out_path = OUT_DIR / combined_name
    combined.to_csv(out_path, index=False)
    LOG.info("Wrote %s (%d rows)", out_path, len(combined))
    return combined


def _write_data_readme(
    topic_mid: str,
    topic_title: str,
    *,
    chatgpt_mid: str = "",
    chatgpt_title: str = GOOGLE_TRENDS_CHATGPT_TOPIC_LABEL,
) -> None:
    """Function summary: write data/raw/circumvention/README.md describing outputs."""
    text = f"""# Circumvention / adaptation raw data

Downloaded by `scripts/download_circumvention_data.py` (Kreitmeir & Raschky 2023 replication).

## Event window

- Ban onset (Italy): {BAN_DATE}
- Ban lifted: {LIFT_DATE}
- Data window: {START} – {END}
- Treated: {TREATED}; controls: {", ".join(CONTROLS)}

## Tor Metrics (`tor/`)

- **Relay** (`userstats-relay-country_*`): estimated daily **direct** Tor clients (excludes bridge users).
- **Bridge** (`userstats-bridge-country_*`): estimated daily clients via **bridge** relays (harder for firewalls to block).
- Units: estimated user counts per day (see [Tor Metrics documentation](https://metrics.torproject.org/)).
- Combined files: `tor_relay_users_by_country.csv`, `tor_bridge_users_by_country.csv` add `query_country` (lowercase ISO-2).

## Google Trends (`google_trends/`)

### VPN topic

- **Query**: Google Trends **topic** "{topic_title}" (`mid={topic_mid}`), not the bare search term "VPN".
- Per-geo files: `gtrends_vpn_{{GEO}}_{START}_{END}.csv`; combined: `google_trends_vpn_by_country.csv`.
- **Units**: `vpn_interest` is Google's 0–100 index for that geo and date window (max day in window = 100).

### ChatGPT topic (attention/salience, not usage)

- **Query**: Google Trends **topic** "{chatgpt_title}" (`mid={chatgpt_mid or '(unresolved)'}`).
- Per-geo files: `gtrends_chatgpt_{{GEO}}_{START}_{END}.csv`; combined: `google_trends_chatgpt_by_country.csv`.
- **Units**: `chatgpt_interest` uses the same 0–100 within-geo scaling.

### Normalization caveat (both series)

Each single-geo query is rescaled **within that country and period**. Levels are **not comparable across countries**; use only **within-country over-time** movement.

## Manual fallback

If Google blocks automated requests (HTTP 429), export daily data from [Google Trends](https://trends.google.com) for the same topic and window, save as:

`google_trends/gtrends_vpn_{{GEO}}_{START}_{END}.csv` or `gtrends_chatgpt_{{GEO}}_{START}_{END}.csv`

with columns `date`, `vpn_interest` or `chatgpt_interest`, then re-run the script.

## Provenance

See `_manifest.json` for URLs, timestamps, row counts, and git commit.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def _write_manifest(
    manifest_sources: list[dict[str, Any]],
    failures: list[str],
    topic_mid: str,
    topic_title: str,
    *,
    chatgpt_mid: str = "",
    chatgpt_title: str = GOOGLE_TRENDS_CHATGPT_TOPIC_LABEL,
) -> None:
    """Function summary: write JSON provenance manifest under OUT_DIR."""
    payload = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "parameters": {
            "START": START,
            "END": END,
            "BAN_DATE": BAN_DATE,
            "LIFT_DATE": LIFT_DATE,
            "TREATED": TREATED,
            "CONTROLS": CONTROLS,
            "COUNTRIES": COUNTRIES,
            "google_trends_vpn_topic_label": GOOGLE_TRENDS_TOPIC_LABEL,
            "google_trends_vpn_topic_mid": topic_mid,
            "google_trends_vpn_topic_title": topic_title,
            "google_trends_chatgpt_topic_label": GOOGLE_TRENDS_CHATGPT_TOPIC_LABEL,
            "google_trends_chatgpt_topic_mid": chatgpt_mid,
            "google_trends_chatgpt_topic_title": chatgpt_title,
            "google_trends_query_type": "topic",
            "related_queries_expansion": False,
            "tor_user_agent": TOR_USER_AGENT,
        },
        "sources": manifest_sources,
        "failures": failures,
    }
    path = OUT_DIR / "_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Wrote %s", path)


def _print_summary(manifest_sources: list[dict[str, Any]], failures: list[str]) -> None:
    """Function summary: print fixed-width summary table to stdout."""
    rows: list[tuple[str, str, str, str, str, str]] = []
    for rec in manifest_sources:
        name = str(rec.get("name", "?"))
        country = str(rec.get("country", "?"))
        err = rec.get("error")
        code = rec.get("status_code")
        if err or (code is not None and code != 200):
            status = "FAIL"
        else:
            status = "OK"
        rows.append(
            (
                name,
                country,
                status,
                str(rec.get("row_count", "")),
                str(rec.get("date_min", "")),
                str(rec.get("date_max", "")),
            )
        )

    header = ("source", "country", "status", "rows", "date_min", "date_max")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print("\n" + fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))

    if failures:
        print("\nFailures (retry or manual export):")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nAll sources succeeded.")


def main() -> int:
    """Function summary: orchestrate Tor and Google Trends downloads and provenance."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOR_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_sources: list[dict[str, Any]] = []
    failures: list[str] = []

    LOG.info("=== Tor Metrics ===")
    download_tor_all(manifest_sources, failures)
    _combine_tor("relay", "tor_relay_users_by_country.csv")
    _combine_tor("bridge", "tor_bridge_users_by_country.csv")

    topic_mid = ""
    topic_title = GOOGLE_TRENDS_TOPIC_LABEL
    chatgpt_mid = ""
    chatgpt_title = GOOGLE_TRENDS_CHATGPT_TOPIC_LABEL
    trends_ok = False
    chatgpt_ok = False
    pytrends: TrendReq | None = None

    LOG.info("=== Google Trends VPN (topic) ===")
    try:
        # retries=0 avoids pytrends Retry (incompatible with urllib3 2.x); we retry in download_google_trends
        pytrends = TrendReq(hl=TRENDS_HL, tz=TRENDS_TZ, retries=0, backoff_factor=0)
        topic_mid, topic_title, _ = _resolve_topic_mid(
            pytrends, GOOGLE_TRENDS_TOPIC_LABEL, GOOGLE_TRENDS_TOPIC_MID
        )
        LOG.info("Resolved VPN topic: %s (%s)", topic_title, topic_mid)
        df_gt = download_google_trends(
            pytrends, topic_mid, topic_title, manifest_sources, failures, series="vpn"
        )
        trends_ok = df_gt is not None and len(df_gt) > 0
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Google Trends VPN block failed: %s", exc)
        failures.append("google_trends_vpn:all")
        manifest_sources.append(
            {
                "name": "google_trends_vpn_resolve",
                "error": str(exc),
                "country": "all",
            }
        )

    LOG.info("=== Google Trends ChatGPT (topic) ===")
    try:
        if pytrends is None:
            pytrends = TrendReq(hl=TRENDS_HL, tz=TRENDS_TZ, retries=0, backoff_factor=0)
        chatgpt_mid, chatgpt_title, _ = _resolve_topic_mid(
            pytrends, GOOGLE_TRENDS_CHATGPT_TOPIC_LABEL, GOOGLE_TRENDS_CHATGPT_TOPIC_MID
        )
        LOG.info("Resolved ChatGPT topic: %s (%s)", chatgpt_title, chatgpt_mid)
        df_cg = download_google_trends(
            pytrends, chatgpt_mid, chatgpt_title, manifest_sources, failures, series="chatgpt"
        )
        chatgpt_ok = df_cg is not None and len(df_cg) > 0
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Google Trends ChatGPT block failed: %s", exc)
        failures.append("google_trends_chatgpt:all")
        manifest_sources.append(
            {
                "name": "google_trends_chatgpt_resolve",
                "error": str(exc),
                "country": "all",
            }
        )

    _write_manifest(
        manifest_sources,
        failures,
        topic_mid,
        topic_title,
        chatgpt_mid=chatgpt_mid,
        chatgpt_title=chatgpt_title,
    )
    _write_data_readme(
        topic_mid or "(unresolved)",
        topic_title,
        chatgpt_mid=chatgpt_mid or "(unresolved)",
        chatgpt_title=chatgpt_title,
    )
    _print_summary(manifest_sources, failures)

    tor_any = any(
        s.get("name", "").startswith("tor_")
        and not s.get("error")
        and s.get("status_code") == 200
        for s in manifest_sources
    )
    if not tor_any and not trends_ok and not chatgpt_ok:
        LOG.error("All sources failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
