"""Analyze ICA debug outputs and recommend a conservative exclude setting.

This script reads the already generated files under
results/preprocessing_ica_debug/. It does not rerun preprocessing and does not
modify preprocess.py. The goal is to decide whether any ICA component exclusion
looks reasonable enough for a course-project preprocessing workflow.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


ICA_DEBUG_DIR = Path("results/preprocessing_ica_debug")
SUMMARY_CSV_PATH = ICA_DEBUG_DIR / "ica_comparison_summary.csv"
REPORT_PATH = ICA_DEBUG_DIR / "ica_analysis_report.txt"
SETTING_DIRS = {
    "exclude=[]": ICA_DEBUG_DIR / "s01_exclude_none",
    "exclude=[0]": ICA_DEBUG_DIR / "s01_exclude_0",
    "exclude=[1]": ICA_DEBUG_DIR / "s01_exclude_1",
    "exclude=[0,1]": ICA_DEBUG_DIR / "s01_exclude_0_1",
}


def parse_float_pair(line: str) -> tuple[float, float]:
    """Parse lines like 'std: 146.211 -> 141.915'."""
    value_part = line.split(":", 1)[1] if ":" in line else line
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", value_part)
    if len(numbers) < 2:
        raise ValueError(f"Cannot parse before/after values from line: {line}")
    return float(numbers[0]), float(numbers[1])


def parse_component_list(line: str) -> list[int]:
    """Parse component lists such as '[0, 1]' from summary lines."""
    match = re.search(r"\[(.*?)\]", line)
    if not match or not match.group(1).strip():
        return []
    return [int(item.strip()) for item in match.group(1).split(",")]


def load_summary(summary_path: Path) -> dict:
    """Read overall ICA metrics and component energy ranking from summary txt."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing ICA summary file: {summary_path}")

    parsed = {
        "component_energy": {},
    }
    in_energy_section = False

    for line in summary_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("removed_components:"):
            parsed["removed_components"] = parse_component_list(line)
        elif line.startswith("mean:"):
            parsed["overall_mean_before"], parsed["overall_mean_after"] = parse_float_pair(line)
        elif line.startswith("std:"):
            parsed["overall_std_before"], parsed["overall_std_after"] = parse_float_pair(line)
        elif line.startswith("var:"):
            parsed["overall_var_before"], parsed["overall_var_after"] = parse_float_pair(line)
        elif line.startswith("30-45 Hz power ratio:"):
            parsed["hf_ratio_before"], parsed["hf_ratio_after"] = parse_float_pair(line)
        elif line.startswith("Component energy ranking:"):
            in_energy_section = True
        elif in_energy_section and line.startswith("component"):
            match = re.match(r"component\s+(\d+):\s+([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", line)
            if match:
                parsed["component_energy"][int(match.group(1))] = float(match.group(2))

    required = [
        "removed_components",
        "overall_std_before",
        "overall_std_after",
        "overall_var_before",
        "overall_var_after",
        "hf_ratio_before",
        "hf_ratio_after",
    ]
    missing = [key for key in required if key not in parsed]
    if missing:
        raise ValueError(f"Missing fields in {summary_path}: {missing}")

    return parsed


def load_trial_channel_stats(csv_path: Path) -> dict:
    """Summarize per trial/channel std changes from debug CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing trial/channel stats CSV: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "trial": int(row["trial"]),
                    "channel": int(row["channel"]),
                    "std_before": float(row["std_before"]),
                    "std_after": float(row["std_after"]),
                    "std_delta": float(row["std_delta"]),
                }
            )

    std_before = np.asarray([row["std_before"] for row in rows])
    std_after = np.asarray([row["std_after"] for row in rows])
    std_delta = np.asarray([row["std_delta"] for row in rows])
    relative_change = np.divide(
        std_after - std_before,
        std_before,
        out=np.zeros_like(std_after),
        where=std_before != 0,
    )

    channel_changes = {}
    for channel in sorted({row["channel"] for row in rows}):
        channel_rows = [row for row in rows if row["channel"] == channel]
        before = np.asarray([row["std_before"] for row in channel_rows])
        after = np.asarray([row["std_after"] for row in channel_rows])
        rel = np.divide(after - before, before, out=np.zeros_like(after), where=before != 0)
        channel_changes[channel] = float(np.mean(rel))

    most_changed_channel = max(
        channel_changes,
        key=lambda channel: abs(channel_changes[channel]),
    )

    return {
        "mean_std_delta": float(np.mean(std_delta)),
        "mean_relative_std_change": float(np.mean(relative_change)),
        "max_abs_relative_std_change": float(np.max(np.abs(relative_change))),
        "collapsed_fraction_20pct": float(np.mean(relative_change <= -0.20)),
        "collapsed_fraction_40pct": float(np.mean(relative_change <= -0.40)),
        "most_changed_channel": most_changed_channel,
        "most_changed_channel_relative_change": channel_changes[most_changed_channel],
    }


def analyze_setting(label: str, directory: Path) -> dict:
    """Combine summary and CSV metrics for one ICA exclude setting."""
    summary = load_summary(directory / "ica_debug_summary.txt")
    channel_stats = load_trial_channel_stats(
        directory / "trial_channel_mean_std_before_after_ica.csv"
    )
    std_change_ratio = (
        (summary["overall_std_after"] - summary["overall_std_before"])
        / summary["overall_std_before"]
    )
    var_change_ratio = (
        (summary["overall_var_after"] - summary["overall_var_before"])
        / summary["overall_var_before"]
    )
    hf_change_ratio = (
        (summary["hf_ratio_after"] - summary["hf_ratio_before"])
        / summary["hf_ratio_before"]
    )

    return {
        "setting": label,
        "directory": directory,
        **summary,
        **channel_stats,
        "overall_std_change_ratio": std_change_ratio,
        "overall_var_change_ratio": var_change_ratio,
        "hf_ratio_change_ratio": hf_change_ratio,
    }


def score_setting(metrics: dict) -> dict:
    """Score one setting conservatively for artifact removal reasonableness."""
    score = 50.0
    reasons = []
    risks = []

    hf_drop = -metrics["hf_ratio_change_ratio"]
    std_drop = -metrics["overall_std_change_ratio"]
    var_drop = -metrics["overall_var_change_ratio"]
    collapsed_20 = metrics["collapsed_fraction_20pct"]
    collapsed_40 = metrics["collapsed_fraction_40pct"]

    if hf_drop > 0.02:
        score += min(15.0, hf_drop * 200)
        reasons.append("30-45 Hz high-frequency ratio decreased.")
    else:
        risks.append("High-frequency ratio did not meaningfully decrease.")

    if 0.01 <= std_drop <= 0.15:
        score += 12.0
        reasons.append("Overall std decreased moderately.")
    elif std_drop > 0.25:
        score -= 25.0
        risks.append("Overall std dropped too much, suggesting possible signal loss.")
    elif std_drop < 0:
        score -= 8.0
        risks.append("Overall std increased after ICA.")

    if 0.02 <= var_drop <= 0.30:
        score += 8.0
        reasons.append("Overall variance decreased in a moderate range.")
    elif var_drop > 0.45:
        score -= 20.0
        risks.append("Overall variance dropped sharply.")

    if collapsed_40 > 0.05:
        score -= 25.0
        risks.append("Some trial/channel std values collapsed by more than 40%.")
    elif collapsed_20 > 0.30:
        score -= 15.0
        risks.append("Many trial/channel std values dropped by more than 20%.")
    else:
        score += 8.0
        reasons.append("Per-channel std changes do not look globally collapsed.")

    if not metrics["removed_components"]:
        score += 5.0
        reasons.append("No component is removed, so risk of deleting EEG signal is lowest.")

    if len(metrics["removed_components"]) >= 2:
        score -= 8.0
        risks.append("Removing multiple components is riskier without strong evidence.")

    score = max(0.0, min(100.0, score))
    grade = "A" if score >= 75 else "B" if score >= 60 else "C" if score >= 45 else "D"

    return {
        "score": score,
        "grade": grade,
        "reasons": reasons,
        "risks": risks,
    }


def infer_component_notes(settings: list[dict]) -> list[str]:
    """Compare component 0 and 1 effects without pretending certainty."""
    by_setting = {item["setting"]: item for item in settings}
    notes = []

    for component, label in [(0, "exclude=[0]"), (1, "exclude=[1]")]:
        metrics = by_setting.get(label)
        if not metrics:
            continue
        hf_drop = -metrics["hf_ratio_change_ratio"]
        std_drop = -metrics["overall_std_change_ratio"]
        energy = metrics["component_energy"].get(component)
        notes.append(
            f"component {component}: high-frequency ratio drop={hf_drop:.2%}, "
            f"overall std drop={std_drop:.2%}, mixing-energy rank value={energy}. "
            "This is only a candidate signal; component time-series and PSD plots "
            "still need human inspection."
        )

    comp0 = by_setting.get("exclude=[0]")
    comp1 = by_setting.get("exclude=[1]")
    if comp0 and comp1:
        comp0_hf = -comp0["hf_ratio_change_ratio"]
        comp1_hf = -comp1["hf_ratio_change_ratio"]
        comp0_std = -comp0["overall_std_change_ratio"]
        comp1_std = -comp1["overall_std_change_ratio"]
        if comp0_hf > comp1_hf and comp0_std <= 0.15:
            notes.append(
                "Component 0 looks slightly more plausible than component 1 as an "
                "artifact candidate by the automatic metrics, but the evidence is "
                "not strong enough to default-delete it without visual review."
            )
        elif comp1_hf > comp0_hf and comp1_std <= 0.15:
            notes.append(
                "Component 1 looks slightly more plausible than component 0 by the "
                "automatic metrics, but the evidence is not strong enough to "
                "default-delete it without visual review."
            )
        else:
            notes.append(
                "The automatic metrics do not clearly separate component 0 and "
                "component 1 as artifact candidates."
            )

    return notes


def save_comparison_csv(settings: list[dict], output_path: Path = SUMMARY_CSV_PATH) -> Path:
    """Save all setting metrics and scores into one comparison table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "setting",
        "removed_components",
        "overall_mean_before",
        "overall_mean_after",
        "overall_std_before",
        "overall_std_after",
        "overall_std_change_ratio",
        "overall_var_before",
        "overall_var_after",
        "overall_var_change_ratio",
        "hf_ratio_before",
        "hf_ratio_after",
        "hf_ratio_change_ratio",
        "mean_relative_std_change",
        "max_abs_relative_std_change",
        "collapsed_fraction_20pct",
        "collapsed_fraction_40pct",
        "most_changed_channel",
        "most_changed_channel_relative_change",
        "score",
        "grade",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in settings:
            writer.writerow(
                {
                    field: item.get(field)
                    for field in fieldnames
                }
            )

    return output_path


def choose_recommendation(settings: list[dict]) -> dict:
    """Choose a conservative recommendation."""
    no_delete = next(item for item in settings if item["setting"] == "exclude=[]")
    best = max(settings, key=lambda item: item["score"])
    improvement = best["score"] - no_delete["score"]
    no_delete_hf_change = abs(no_delete["hf_ratio_change_ratio"])
    best_extra_hf_drop = (
        -best["hf_ratio_change_ratio"]
        - (-no_delete["hf_ratio_change_ratio"])
    )

    if (
        best["setting"] != "exclude=[]"
        and improvement >= 15
        and best_extra_hf_drop >= 0.05
        and best["grade"] in {"A", "B"}
        and best["collapsed_fraction_20pct"] <= 0.05
        and no_delete_hf_change < abs(best["hf_ratio_change_ratio"])
    ):
        return {
            "setting": best["setting"],
            "reason": (
                "This setting scored clearly better than no deletion while avoiding "
                "large global collapse. Manual plot review is still required."
            ),
        }

    return {
        "setting": "exclude=[]",
        "reason": (
            "Current automatic evidence is not strong enough to default-delete an "
            "ICA component. Conservative choice is to keep no component removed "
            "until component time-series/PSD plots provide clearer artifact evidence."
        ),
    }


def save_report(
    settings: list[dict],
    recommendation: dict,
    component_notes: list[str],
    output_path: Path = REPORT_PATH,
) -> Path:
    """Write the ICA analysis report."""
    lines = [
        "ICA debug automatic analysis report",
        "",
        "Important stance:",
        "This report is conservative. It looks for reasonable artifact removal "
        "without strong evidence of EEG signal damage. It does not replace human "
        "inspection of component maps/time series/PSD plots.",
        "",
        "Four-setting overview:",
    ]

    for item in settings:
        lines.extend(
            [
                "",
                f"{item['setting']} | grade={item['grade']} | score={item['score']:.1f}",
                f"removed_components: {item['removed_components']}",
                f"std change: {item['overall_std_change_ratio']:.2%}",
                f"var change: {item['overall_var_change_ratio']:.2%}",
                f"30-45 Hz ratio change: {item['hf_ratio_change_ratio']:.2%}",
                f"mean per trial/channel std change: {item['mean_relative_std_change']:.2%}",
                f"max abs trial/channel std change: {item['max_abs_relative_std_change']:.2%}",
                f"most changed channel: {item['most_changed_channel']} "
                f"({item['most_changed_channel_relative_change']:.2%})",
                "advantages: " + ("; ".join(item["reasons"]) or "none detected"),
                "risks: " + ("; ".join(item["risks"]) or "none detected"),
            ]
        )

    lines.extend(
        [
            "",
            "Component 0/1 notes:",
            *component_notes,
            "",
            "Recommendation:",
            f"Recommended setting: {recommendation['setting']}",
            recommendation["reason"],
            "",
            "Uncertainty:",
            "At present, the evidence is insufficient to claim that any single "
            "component is definitely artifact. The automatic metrics can show "
            "whether exclusion changes scale and high-frequency energy, but they "
            "cannot prove eye movement, muscle artifact, or valid EEG source.",
            "",
            "Next manual checks:",
            "- Inspect component time series for slow large waves, spikes, or high-frequency jitter.",
            "- Inspect PSD before/after for selective artifact reduction rather than broad signal loss.",
            "- Compare several trials/channels, not only trial 0 channel 0.",
            "- If possible, add EOG-related channels or topographic maps before default exclusion.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    """Analyze all generated ICA debug result folders."""
    settings = []
    for label, directory in SETTING_DIRS.items():
        metrics = analyze_setting(label, directory)
        metrics.update(score_setting(metrics))
        settings.append(metrics)

    csv_path = save_comparison_csv(settings)
    recommendation = choose_recommendation(settings)
    component_notes = infer_component_notes(settings)
    report_path = save_report(settings, recommendation, component_notes)

    print(f"saved comparison summary: {csv_path}")
    print(f"saved analysis report: {report_path}")
    print(f"recommended setting: {recommendation['setting']}")
    print(recommendation["reason"])


if __name__ == "__main__":
    main()
