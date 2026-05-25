"""
Script summary:
Export curated italian_political_lexicon_v4.csv into runtime config/lexicons/*.txt files.

Functionality:
- Merges v4 Italian lemmas into political_it.txt, ideology_it.txt (dominant L/C/R default),
  issue_it.txt, other_side_it.txt, pairs_it.json, stance/valence/polarized lexicons, term_meta_it.json.
- Auto-archives ideology_it.txt to ideology_it_broad.txt on first dominant export.
- Writes audit and diff tables under results/tables/italy_polarization/lexicon_export/.

How to apply/run:
  .venv/bin/python scripts/devtools/export_italian_lexicon_v4.py --policy dominant
  .venv/bin/python scripts/features/compute_polarization_features.py --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ISSUE_TOPIC_MAP = {
    "eu": "eu",
    "migration": "migration",
    "labor": "economy",
    "welfare": "economy",
    "fiscal": "economy",
    "economy": "economy",
    "social": "culture",
    "identity": "culture",
    "populism": "culture",
    "law_order": "culture",
    "foreign": "culture",
}

OTHER_SIDE_TOPICS = frozenset({"identity", "populism", "law_order", "social"})


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root()

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.v4_lexicon import dominant_side_for_role, use_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI arguments for v4 lexicon export."""
    parser = argparse.ArgumentParser(description="Export italian_political_lexicon_v4 to config/lexicons/.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to v4 CSV (default: data/raw/italian_political_lexicon_v4.csv)",
    )
    parser.add_argument(
        "--policy",
        choices=("dominant", "broad", "conservative"),
        default="dominant",
        help="dominant=single L/C/R per term; broad=yes/some/rarely; conservative=yes only",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write audit/diff only, do not update lexicon files")
    return parser.parse_args()


def _normalize_use(value: str) -> str:
    """Function summary: normalize a use-cell string for comparison."""
    return (value or "").strip().lower()


def _use_matches(value: str, policy: str) -> bool:
    """Function summary: return whether a use column qualifies under export policy.

    Parameters:
    - value: raw cell (yes, some, rarely, no, etc.).
    - policy: broad or conservative.

    Returns:
    - True if the value assigns the term to that side/category.
    """
    v = _normalize_use(value)
    if policy == "conservative":
        return v == "yes"
    return v in {"yes", "some", "rarely"}


def _norm_term(term: str) -> str:
    """Function summary: lowercase and strip a lemma for deduplication."""
    return (term or "").strip().lower()


def _load_flat_lexicon(path: Path) -> Tuple[List[str], Set[str]]:
    """Function summary: load flat political lexicon lines preserving order.

    Parameters:
    - path: political_it.txt path.

    Returns:
    - Tuple of (ordered lines, set of normalized term keys for dedupe).
    """
    if not path.is_file():
        return [], set()
    lines: List[str] = []
    keys: Set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            lines.append(line)
            continue
        key = _norm_term(line.split(":", 1)[-1] if ":" in line else line)
        if key and key not in keys:
            keys.add(key)
            lines.append(line)
        elif key in keys:
            continue
        else:
            lines.append(line)
    return lines, keys


def _load_categorized_lexicon(path: Path) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Function summary: load categorized lexicon lines and keys per category.

    Parameters:
    - path: ideology_it.txt or issue_it.txt path.

    Returns:
    - Tuple of (ordered lines, category -> normalized term keys).
    """
    if not path.is_file():
        return [], defaultdict(set)
    lines: List[str] = []
    by_cat: Dict[str, Set[str]] = defaultdict(set)
    global_keys: Set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            lines.append(line)
            continue
        if ":" not in line:
            lines.append(line)
            continue
        cat, term = line.split(":", 1)
        cat = cat.strip().lower()
        term_key = _norm_term(term)
        dedupe_key = f"{cat}:{term_key}"
        if term_key and dedupe_key not in global_keys:
            global_keys.add(dedupe_key)
            by_cat[cat].add(term_key)
            lines.append(line)
    return lines, by_cat


def _read_v4_rows(path: Path) -> List[Dict[str, str]]:
    """Function summary: load v4 CSV rows with semicolon delimiter and UTF-8 BOM.

    Parameters:
    - path: italian_political_lexicon_v4.csv path.

    Returns:
    - List of row dicts.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _iter_row_terms(row: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """Function summary: yield (term, role, section) lemmas from one v4 row.

    Parameters:
    - row: CSV dict.

    Returns:
    - List of (lemma, role, section) where role is term_a, term_b, or term.
    """
    section = (row.get("section") or "").strip()
    out: List[Tuple[str, str, str]] = []
    if section == "pairs":
        if row.get("term_a", "").strip():
            out.append((row["term_a"].strip(), "term_a", section))
        if row.get("term_b", "").strip():
            out.append((row["term_b"].strip(), "term_b", section))
    else:
        term = (row.get("term") or row.get("term_a") or "").strip()
        if term:
            out.append((term, "term", section))
    return out


def _ideology_sides_for_term(
    row: Dict[str, str],
    role: str,
    policy: str,
) -> List[str]:
    """Function summary: map v4 use columns to ideology categories for one lemma role.

    Parameters:
    - row: CSV dict.
    - role: term_a, term_b, or term.
    - policy: broad or conservative.

    Returns:
    - List of side names among left, center, right (may be empty or multiple).
    """
    if role == "term_a":
        left_col, center_col, right_col = "term_a_left_use", "term_a_center_use", "term_a_right_use"
    elif role == "term_b":
        left_col, center_col, right_col = "term_b_left_use", "term_b_center_use", "term_b_right_use"
    else:
        left_col, center_col, right_col = "left_use", "center_use", "right_use"
    sides: List[str] = []
    if _use_matches(row.get(left_col, ""), policy):
        sides.append("left")
    if _use_matches(row.get(center_col, ""), policy):
        sides.append("center")
    if _use_matches(row.get(right_col, ""), policy):
        sides.append("right")
    return sides


def _stance_for_role(row: Dict[str, str], role: str) -> str:
    """Function summary: return stance string for a term role in a v4 row."""
    if role == "term_a":
        return _normalize_use(row.get("term_a_stance", "") or row.get("stance", ""))
    if role == "term_b":
        return _normalize_use(row.get("term_b_stance", "") or row.get("stance", ""))
    return _normalize_use(row.get("stance", ""))


def _qualifies_other_side(row: Dict[str, str], role: str, term: str) -> bool:
    """Function summary: decide if a lemma should be added to other_side_it.txt.

    Parameters:
    - row: CSV dict.
    - role: term_a, term_b, or term.
    - term: lemma text.

    Returns:
    - True when stance is contra and topic/term fit other-side salience rules.
    """
    stance = _stance_for_role(row, role)
    if "contra" not in stance:
        return False
    topic = _normalize_use(row.get("topic", ""))
    if topic in OTHER_SIDE_TOPICS:
        return True
    polarized = _normalize_use(row.get("polarized", ""))
    if polarized == "yes":
        return True
    t = _norm_term(term)
    adversary_hints = (
        "fascist",
        "comunist",
        "grillin",
        "leghist",
        "terron",
        "burin",
        "ignorant",
        "parassit",
        "vendut",
        "idiot",
        "cretin",
    )
    return any(h in t for h in adversary_hints)


def _issue_category(topic: str) -> Optional[str]:
    """Function summary: map v4 topic to issue_it category or None."""
    return ISSUE_TOPIC_MAP.get(_normalize_use(topic))


def _archive_broad_ideology_if_needed(ideology_path: Path, broad_path: Path) -> None:
    """Function summary: copy ideology_it.txt to ideology_it_broad.txt before dominant overwrite.

    Parameters:
    - ideology_path: current ideology_it.txt.
    - broad_path: archive destination.

    Returns:
    - None.
    """
    if broad_path.is_file():
        return
    if ideology_path.is_file():
        shutil.copy2(ideology_path, broad_path)
        print(f"[export] archived broad ideology -> {broad_path}", flush=True)


def _stance_bucket(stance: str) -> str:
    """Function summary: map raw stance to pro/contra/ambiguous category."""
    s = _normalize_use(stance)
    if "contra" in s:
        return "contra"
    if s == "pro":
        return "pro"
    return "ambiguous"


def _valence_bucket(valence: str) -> str:
    """Function summary: map raw valence to positive/negative/neutral/ambiguous."""
    v = _normalize_use(valence)
    if v in {"positive", "negative", "neutral"}:
        return v
    return "ambiguous"


def _ideology_side_for_export(row: Dict[str, str], role: str, policy: str) -> Optional[str]:
    """Function summary: return single ideology side for export under policy."""
    if policy == "dominant":
        side, _rule = dominant_side_for_role(row, role, variant="dominant_v1")
        return side
    sides = _ideology_sides_for_term(row, role, policy)
    return sides[0] if len(sides) == 1 else None


def _collect_v4_term_keys(rows: List[Dict[str, str]]) -> Set[str]:
    """Function summary: normalized keys for all lemmas in v4 CSV."""
    keys: Set[str] = set()
    for row in rows:
        for term, _role, _sec in _iter_row_terms(row):
            k = _norm_term(term)
            if k:
                keys.add(k)
    return keys


def _filter_ideology_lines_without_v4(
    old_lines: List[str],
    v4_keys: Set[str],
) -> List[str]:
    """Function summary: drop categorized lines whose term appears in v4 (for dominant rebuild)."""
    kept: List[str] = []
    for raw in old_lines:
        line = raw.rstrip()
        if not line or line.startswith("#"):
            kept.append(line)
            continue
        if ":" not in line:
            kept.append(line)
            continue
        _cat, term = line.split(":", 1)
        if _norm_term(term) in v4_keys:
            continue
        kept.append(line)
    return kept


def export_lexicons(
    root: Path,
    input_path: Path,
    policy: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Function summary: run v4 export and write lexicon files plus audit tables.

    Parameters:
    - root: repository root.
    - input_path: v4 CSV path.
    - policy: broad or conservative.
    - dry_run: if True, skip writing lexicon txt files.

    Returns:
    - Summary dict with counts.
    """
    lex_dir = root / "config" / "lexicons"
    tables_dir = root / "results" / "tables" / "italy_polarization"
    tables_dir.mkdir(parents=True, exist_ok=True)

    political_path = lex_dir / "political_it.txt"
    ideology_path = lex_dir / "ideology_it.txt"
    issue_path = lex_dir / "issue_it.txt"
    other_side_path = lex_dir / "other_side_it.txt"

    old_political_lines, old_political_keys = _load_flat_lexicon(political_path)
    old_ideology_lines, old_ideology_by_cat = _load_categorized_lexicon(ideology_path)
    old_issue_lines, old_issue_by_cat = _load_categorized_lexicon(issue_path)
    old_other_lines, old_other_by_cat = _load_categorized_lexicon(other_side_path)

    rows = _read_v4_rows(input_path)
    v4_keys = _collect_v4_term_keys(rows)
    political_keys = set(old_political_keys)
    if policy == "dominant":
        old_ideology_lines, _ = _load_categorized_lexicon(ideology_path)
        old_ideology_lines = _filter_ideology_lines_without_v4(old_ideology_lines, v4_keys)
        old_ideology_by_cat = defaultdict(set)
    ideology_by_cat: Dict[str, Set[str]] = {k: set(v) for k, v in old_ideology_by_cat.items()}
    issue_by_cat: Dict[str, Set[str]] = {k: set(v) for k, v in old_issue_by_cat.items()}
    other_keys = set(old_other_by_cat.get("other_side", set()))

    new_political: List[str] = []
    new_ideology: List[Tuple[str, str]] = []  # (category, term)
    new_issue: List[Tuple[str, str]] = []
    new_other: List[str] = []
    audit_rows: List[Dict[str, str]] = []
    pairs_out: List[Dict[str, Any]] = []
    term_meta: Dict[str, Dict[str, Any]] = {}
    stance_by_cat: Dict[str, Set[str]] = defaultdict(set)
    valence_by_cat: Dict[str, Set[str]] = defaultdict(set)
    polarized_by_cat: Dict[str, Set[str]] = defaultdict(set)
    pair_id = 0

    for row in rows:
        topic = (row.get("topic") or "").strip()
        section = (row.get("section") or "").strip()
        issue_cat = _issue_category(topic)

        for term, role, sec in _iter_row_terms(row):
            term_display = term.strip()
            term_key = _norm_term(term_display)
            if not term_key:
                continue

            assignments: List[str] = []
            skipped: List[str] = []

            if term_key not in political_keys:
                political_keys.add(term_key)
                new_political.append(term_display)
                assignments.append("political_it")
            else:
                skipped.append("political_it:duplicate")

            if policy == "dominant":
                side, tie_rule = dominant_side_for_role(row, role, variant="dominant_v1")
                if side and term_key not in ideology_by_cat.get(side, set()):
                    ideology_by_cat.setdefault(side, set()).add(term_key)
                    new_ideology.append((side, term_display))
                    assignments.append(f"ideology_it:{side}")
                elif not side:
                    skipped.append(f"ideology_it:dominant_omit:{tie_rule}")
                else:
                    skipped.append(f"ideology_it:{side}:duplicate")
            else:
                sides = _ideology_sides_for_term(row, role, policy)
                for side in sides:
                    if term_key not in ideology_by_cat.get(side, set()):
                        ideology_by_cat.setdefault(side, set()).add(term_key)
                        new_ideology.append((side, term_display))
                        assignments.append(f"ideology_it:{side}")
                    else:
                        skipped.append(f"ideology_it:{side}:duplicate")

            if policy == "dominant":
                if role == "term_a":
                    lu, cu, ru = row.get("term_a_left_use", ""), row.get("term_a_center_use", ""), row.get("term_a_right_use", "")
                    stance_v = row.get("term_a_stance", "") or row.get("stance", "")
                    valence_v = row.get("term_a_valence", "") or row.get("valence", "")
                elif role == "term_b":
                    lu, cu, ru = row.get("term_b_left_use", ""), row.get("term_b_center_use", ""), row.get("term_b_right_use", "")
                    stance_v = row.get("term_b_stance", "") or row.get("stance", "")
                    valence_v = row.get("term_b_valence", "") or row.get("valence", "")
                else:
                    lu, cu, ru = row.get("left_use", ""), row.get("center_use", ""), row.get("right_use", "")
                    stance_v = row.get("stance", "")
                    valence_v = row.get("valence", "")
                dom, tie_rule = dominant_side_for_role(row, role, variant="dominant_v1")
                term_meta[term_key] = {
                    "dominant_side": dom,
                    "tie_rule": tie_rule,
                    "use_scores": {"left": use_score(lu), "center": use_score(cu), "right": use_score(ru)},
                    "stance": _stance_bucket(stance_v),
                    "valence": _valence_bucket(valence_v),
                    "polarized": _normalize_use(row.get("polarized", "")),
                    "political_relevance": int((row.get("political_relevance") or "0").strip() or 0),
                }
                sb = _stance_bucket(stance_v)
                vb = _valence_bucket(valence_v)
                if term_key not in stance_by_cat[sb]:
                    stance_by_cat[sb].add(term_key)
                if term_key not in valence_by_cat[vb]:
                    valence_by_cat[vb].add(term_key)
                pol_flag = _normalize_use(row.get("polarized", ""))
                if pol_flag in {"yes", "no"} and term_key not in polarized_by_cat[pol_flag]:
                    polarized_by_cat[pol_flag].add(term_key)

            if issue_cat:
                if term_key not in issue_by_cat.get(issue_cat, set()):
                    issue_by_cat.setdefault(issue_cat, set()).add(term_key)
                    new_issue.append((issue_cat, term_display))
                    assignments.append(f"issue_it:{issue_cat}")
                else:
                    skipped.append(f"issue_it:{issue_cat}:duplicate")
            else:
                skipped.append("issue_it:topic_unmapped")

            if _qualifies_other_side(row, role, term_display):
                if term_key not in other_keys:
                    other_keys.add(term_key)
                    new_other.append(term_display)
                    assignments.append("other_side_it")
                else:
                    skipped.append("other_side_it:duplicate")

            audit_rows.append(
                {
                    "term": term_display,
                    "section": section,
                    "role": role,
                    "topic": topic,
                    "policy": policy,
                    "assignments": ";".join(assignments) if assignments else "",
                    "skipped": ";".join(skipped) if skipped else "",
                    "dominant_side": term_meta.get(term_key, {}).get("dominant_side", ""),
                    "tie_rule": term_meta.get(term_key, {}).get("tie_rule", ""),
                }
            )

        if section == "pairs" and policy == "dominant":
            ta = (row.get("term_a") or "").strip()
            tb = (row.get("term_b") or "").strip()
            if ta and tb:
                pa, _ = dominant_side_for_role(row, "term_a", variant="dominant_v1")
                pb, _ = dominant_side_for_role(row, "term_b", variant="dominant_v1")
                pairs_out.append(
                    {
                        "pair_id": f"{topic}_{pair_id}",
                        "topic": topic,
                        "term_a": ta,
                        "term_b": tb,
                        "pole_a": pa or "ambiguous",
                        "pole_b": pb or "ambiguous",
                        "polarized": _normalize_use(row.get("polarized", "")),
                    }
                )
                pair_id += 1

    diff_rows: List[Dict[str, str]] = []
    for t in new_political:
        diff_rows.append({"file": "political_it.txt", "category": "", "term": t, "action": "added"})
    for cat, t in new_ideology:
        diff_rows.append({"file": "ideology_it.txt", "category": cat, "term": t, "action": "added"})
    for cat, t in new_issue:
        diff_rows.append({"file": "issue_it.txt", "category": cat, "term": t, "action": "added"})
    for t in new_other:
        diff_rows.append({"file": "other_side_it.txt", "category": "other_side", "term": t, "action": "added"})

    export_dir = tables_dir / "lexicon_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    audit_path = export_dir / "lexicon_v4_export_audit.csv"
    diff_path = export_dir / "lexicon_v4_export_diff.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "term",
                "section",
                "role",
                "topic",
                "policy",
                "assignments",
                "skipped",
                "dominant_side",
                "tie_rule",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    with diff_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "category", "term", "action"])
        writer.writeheader()
        writer.writerows(diff_rows)

    if not dry_run:
        if policy == "dominant":
            _archive_broad_ideology_if_needed(
                ideology_path,
                lex_dir / "ideology_it_broad.txt",
            )
        _write_political_it(political_path, old_political_lines, new_political, input_path, policy)
        if policy == "dominant":
            _write_ideology_it_dominant(ideology_path, old_ideology_lines, new_ideology, input_path, policy)
        else:
            _write_ideology_it(ideology_path, old_ideology_lines, new_ideology, input_path, policy)
        _write_issue_it(issue_path, old_issue_lines, new_issue, input_path, policy)
        _write_other_side_it(other_side_path, old_other_lines, new_other, input_path, policy)
        if policy == "dominant":
            _write_pairs_json(lex_dir / "pairs_it.json", pairs_out, input_path)
            _write_term_meta_json(lex_dir / "term_meta_it.json", term_meta, input_path)
            _write_categorized_flat(lex_dir / "stance_it.txt", stance_by_cat, "stance", input_path, policy)
            _write_categorized_flat(lex_dir / "valence_it.txt", valence_by_cat, "valence", input_path, policy)
            _write_categorized_flat(lex_dir / "polarized_it.txt", polarized_by_cat, "polarized", input_path, policy)
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            (lex_dir / "dominant_export_stamp.txt").write_text(
                f"ideology_scoring=dominant_v1\nexported_at_utc={stamp}\nsource={input_path.name}\n",
                encoding="utf-8",
            )

    summary = {
        "v4_rows": len(rows),
        "audit_rows": len(audit_rows),
        "added_political": len(new_political),
        "added_ideology": len(new_ideology),
        "added_issue": len(new_issue),
        "added_other_side": len(new_other),
        "audit_path": str(audit_path),
        "diff_path": str(diff_path),
        "dry_run": dry_run,
    }
    return summary


def _write_political_it(
    path: Path,
    old_lines: List[str],
    new_terms: List[str],
    input_path: Path,
    policy: str,
) -> None:
    """Function summary: write merged political_it.txt with v4 provenance header."""
    body = list(old_lines)
    v4_meta = f"# v4 export ({policy}) from {input_path.name}"
    if not any("v4 export" in ln for ln in body[:8]):
        insert_at = 1 if body and body[0].startswith("#") else 0
        body.insert(insert_at, v4_meta)
    if new_terms:
        if not any("v4 additions" in ln for ln in body):
            body.append(f"# v4 additions ({policy})")
        body.extend(new_terms)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _write_categorized_append(
    path: Path,
    old_lines: List[str],
    new_entries: List[Tuple[str, str]],
    category_prefix: str,
    header_comment: str,
    input_path: Path,
    policy: str,
) -> None:
    """Function summary: append categorized lines to an ideology/issue/other_side file."""
    lines = list(old_lines)
    if not any("v4 export" in ln for ln in lines[:5]):
        insert_at = 0
        for i, ln in enumerate(lines):
            if ln.startswith("#") or not ln.strip():
                insert_at = i + 1
            else:
                break
        lines.insert(insert_at, f"# v4 export ({policy}) from {input_path.name} — {header_comment}")
    if new_entries:
        if not any("v4 additions" in ln for ln in lines):
            lines.append(f"# v4 additions ({policy})")
        for cat, term in sorted(new_entries, key=lambda x: (x[0], x[1].lower())):
            lines.append(f"{category_prefix}{cat}:{term}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ideology_it(
    path: Path,
    old_lines: List[str],
    new_entries: List[Tuple[str, str]],
    input_path: Path,
    policy: str,
) -> None:
    """Function summary: write ideology_it.txt with v4 left/center/right entries."""
    _write_categorized_append(
        path,
        old_lines,
        new_entries,
        "",
        "ideology L/C/R from use columns",
        input_path,
        policy,
    )


def _write_ideology_it_dominant(
    path: Path,
    old_lines: List[str],
    new_entries: List[Tuple[str, str]],
    input_path: Path,
    policy: str,
) -> None:
    """Function summary: rewrite ideology_it.txt with dominant single-side v4 terms."""
    lines = list(old_lines)
    header = f"# dominant export ({policy}) from {input_path.name} — one L/C/R per v4 term"
    if not any("dominant export" in ln for ln in lines[:8]):
        lines.insert(0, header)
    if new_entries:
        lines.append(f"# v4 dominant additions ({policy})")
        for cat, term in sorted(new_entries, key=lambda x: (x[0], x[1].lower())):
            lines.append(f"{cat}:{term}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pairs_json(path: Path, pairs: List[Dict[str, Any]], input_path: Path) -> None:
    """Function summary: write pairs_it.json registry."""
    payload = {
        "source": input_path.name,
        "ideology_scoring": "dominant_v1",
        "pairs": pairs,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_term_meta_json(path: Path, meta: Dict[str, Dict[str, Any]], input_path: Path) -> None:
    """Function summary: write term_meta_it.json for weighted/metadata scoring."""
    payload = {"source": input_path.name, "terms": meta}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_categorized_flat(
    path: Path,
    by_cat: Dict[str, Set[str]],
    family: str,
    input_path: Path,
    policy: str,
) -> None:
    """Function summary: write a categorized lexicon file from category->term keys."""
    lines = [f"# v4 {family} export ({policy}) from {input_path.name}"]
    for cat in sorted(by_cat.keys()):
        for term_key in sorted(by_cat[cat]):
            lines.append(f"{cat}:{' '.join(term_key.split())}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_issue_it(
    path: Path,
    old_lines: List[str],
    new_entries: List[Tuple[str, str]],
    input_path: Path,
    policy: str,
) -> None:
    """Function summary: write issue_it.txt with v4 topic-mapped entries."""
    _write_categorized_append(
        path,
        old_lines,
        new_entries,
        "",
        "issue salience from mapped topics",
        input_path,
        policy,
    )


def _write_other_side_it(
    path: Path,
    old_lines: List[str],
    new_terms: List[str],
    input_path: Path,
    policy: str,
) -> None:
    """Function summary: write other_side_it.txt with v4 contra/adversary terms."""
    entries = [("other_side", t) for t in new_terms]
    _write_categorized_append(
        path,
        old_lines,
        entries,
        "",
        "other-side salience (stance=contra)",
        input_path,
        policy,
    )


def main() -> None:
    """Function summary: CLI entrypoint for v4 lexicon export."""
    args = parse_args()
    root = PROJECT_ROOT
    input_path = args.input or (root / "data/raw/italian_political_lexicon_v4.csv")
    if not input_path.is_file():
        raise FileNotFoundError(f"v4 lexicon not found: {input_path}")
    summary = export_lexicons(root, input_path, args.policy, dry_run=args.dry_run)
    print("[export_italian_lexicon_v4] done", flush=True)
    for key, val in summary.items():
        print(f"  {key}: {val}", flush=True)


if __name__ == "__main__":
    main()
