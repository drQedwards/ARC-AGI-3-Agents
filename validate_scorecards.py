#!/usr/bin/env python3
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def probe_scorecard(card_id: str) -> tuple[bool, str]:
    url = f"https://three.arcprize.org/scorecards/{card_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return True, url
        return False, f"{url} -> HTTP {r.status_code}"
    except Exception as exc:
        return False, f"{url} -> {exc.__class__.__name__}: {exc}"


def main() -> int:
    option_a = load_json(ROOT / "scorecard_option_a_normal.json")
    option_d = load_json(ROOT / "scorecard_option_d_offline.json")

    lines = ["# Scorecard Validation Report", ""]

    if option_a:
        b = option_a["baseline"]
        w = option_a["worldmodel"]
        lines += [
            "## Option A (normal + fallback)",
            f"- Baseline completion: {b['levels_completed']}/{b['total_levels']} ({b['completion_rate']})",
            f"- Worldmodel completion: {w['levels_completed']}/{w['total_levels']} ({w['completion_rate']})",
        ]
        for label, card in [("baseline", b["card_id"]), ("worldmodel", w["card_id"])]:
            ok, msg = probe_scorecard(card)
            state = "visible" if ok else "not visible"
            lines.append(f"- Online scorecard ({label}) is {state}: {msg}")
        lines.append("")
    else:
        lines += ["## Option A", "- scorecard_option_a_normal.json not found", ""]

    if option_d:
        lines += [
            "## Option D (offline)",
            f"- Baseline win rate: {option_d['baseline']['win_rate']}",
            f"- Worldmodel win rate: {option_d['worldmodel']['win_rate']}",
            "- This mode is local/offline only, so no online scorecard URL is expected.",
            "",
        ]

    lines += [
        "## Kaggle submission steps (if online score is missing)",
        "1. Open `arc_agi3_comparison_kaggle.ipynb` in Kaggle.",
        "2. Add your `OPENAI_API_KEY` and `ARC_API_KEY` in Kaggle Secrets.",
        "3. Run all cells for Option C (online sweep) to generate official scorecards.",
        "4. Copy the resulting scorecard URL(s) from cell output.",
        "5. Submit the score via the contest form: https://forms.gle/wMLZrEFGDh33DhzV9.",
    ]

    out = ROOT / "scorecard_validation_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
