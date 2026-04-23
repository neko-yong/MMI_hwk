"""Minimal active learning experiment for single-subject DEAP classification.

This module is intentionally small and course-assignment oriented. It uses only
subject s01 with 40 trials, so the results are a sanity-check comparison between
random sampling and uncertainty sampling rather than a strong conclusion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.train_baseline import (
    DEFAULT_SUBJECT_ID,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RANDOM_STATE,
    build_single_subject_feature_matrix,
)
from src.utils import set_random_seed


INITIAL_LABELED_RATIO = 0.20
TEST_SIZE = 0.25
QUERY_SIZE = 2
RESULTS_DIR = Path("results")
ACTIVE_LEARNING_ACCURACY_FIGURE_PATH = (
    RESULTS_DIR / "active_learning_s01_accuracy.png"
)
ACTIVE_LEARNING_F1_FIGURE_PATH = RESULTS_DIR / "active_learning_s01_f1.png"


try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_svm_classifier(random_state: int = RANDOM_STATE) -> Pipeline:
    """Create the same standardized RBF SVM used for active learning rounds."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    random_state=random_state,
                ),
            ),
        ]
    )


def ensure_initial_labeled_set_has_both_classes(
    labels: np.ndarray,
    candidate_indices: np.ndarray,
    initial_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a small initial labeled set containing both classes if possible."""
    selected = []

    for label in [NEGATIVE_LABEL, POSITIVE_LABEL]:
        label_candidates = candidate_indices[labels[candidate_indices] == label]
        if len(label_candidates) > 0:
            selected.append(int(rng.choice(label_candidates)))

    remaining = np.asarray(
        [index for index in candidate_indices if index not in selected],
        dtype=np.int64,
    )
    rng.shuffle(remaining)

    needed = max(0, initial_count - len(selected))
    selected.extend(int(index) for index in remaining[:needed])

    return np.asarray(selected, dtype=np.int64)


def initialize_active_learning_split(
    labels: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into initial labeled set, unlabeled pool, and fixed test set."""
    all_indices = np.arange(len(labels))
    pool_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=labels,
    )

    initial_count = max(2, int(round(len(pool_indices) * INITIAL_LABELED_RATIO)))
    rng = np.random.default_rng(random_state)
    labeled_indices = ensure_initial_labeled_set_has_both_classes(
        labels,
        pool_indices,
        initial_count,
        rng,
    )
    unlabeled_indices = np.asarray(
        [index for index in pool_indices if index not in set(labeled_indices)],
        dtype=np.int64,
    )

    return labeled_indices, unlabeled_indices, test_indices


def evaluate_classifier(
    classifier: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    test_indices: np.ndarray,
) -> dict:
    """Evaluate the current classifier on the fixed test set."""
    y_true = labels[test_indices]
    y_pred = classifier.predict(features[test_indices])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def select_random_samples(
    unlabeled_indices: np.ndarray,
    query_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select samples uniformly at random from the unlabeled pool."""
    selected_count = min(query_size, len(unlabeled_indices))
    return rng.choice(unlabeled_indices, size=selected_count, replace=False)


def select_uncertain_samples(
    classifier: Pipeline,
    features: np.ndarray,
    unlabeled_indices: np.ndarray,
    query_size: int,
) -> np.ndarray:
    """Select samples closest to the SVM decision boundary.

    For binary SVMs, smaller absolute decision_function values mean the sample
    is closer to the boundary and therefore more uncertain.
    """
    selected_count = min(query_size, len(unlabeled_indices))
    decision_values = classifier.decision_function(features[unlabeled_indices])
    uncertainty_order = np.argsort(np.abs(decision_values))
    return unlabeled_indices[uncertainty_order[:selected_count]]


def run_sampling_strategy(
    features: np.ndarray,
    labels: np.ndarray,
    strategy: str,
    initial_labeled_indices: np.ndarray,
    initial_unlabeled_indices: np.ndarray,
    test_indices: np.ndarray,
    query_size: int = QUERY_SIZE,
    random_state: int = RANDOM_STATE,
) -> list[dict]:
    """Run one active learning strategy and return per-round metrics."""
    rng = np.random.default_rng(random_state)
    labeled_indices = initial_labeled_indices.copy()
    unlabeled_indices = initial_unlabeled_indices.copy()
    history = []
    round_index = 0

    while True:
        classifier = create_svm_classifier(random_state=random_state)
        classifier.fit(features[labeled_indices], labels[labeled_indices])
        metrics = evaluate_classifier(classifier, features, labels, test_indices)

        history.append(
            {
                "strategy": strategy,
                "round": round_index,
                "labeled_samples": len(labeled_indices),
                **metrics,
            }
        )

        if len(unlabeled_indices) == 0:
            break

        if strategy == "random":
            selected = select_random_samples(unlabeled_indices, query_size, rng)
        elif strategy == "uncertainty":
            selected = select_uncertain_samples(
                classifier,
                features,
                unlabeled_indices,
                query_size,
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        labeled_indices = np.concatenate([labeled_indices, selected])
        selected_set = set(int(index) for index in selected)
        unlabeled_indices = np.asarray(
            [index for index in unlabeled_indices if int(index) not in selected_set],
            dtype=np.int64,
        )
        round_index += 1

    return history


def run_active_learning_experiment(
    subject_id: int = DEFAULT_SUBJECT_ID,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Run random sampling vs uncertainty sampling on one DEAP subject."""
    features, labels = build_single_subject_feature_matrix(subject_id)
    labeled_indices, unlabeled_indices, test_indices = initialize_active_learning_split(
        labels,
        random_state=random_state,
    )

    random_history = run_sampling_strategy(
        features,
        labels,
        "random",
        labeled_indices,
        unlabeled_indices,
        test_indices,
        random_state=random_state,
    )
    uncertainty_history = run_sampling_strategy(
        features,
        labels,
        "uncertainty",
        labeled_indices,
        unlabeled_indices,
        test_indices,
        random_state=random_state,
    )
    curve_data = build_curve_data(random_history, uncertainty_history)

    return {
        "subject_id": subject_id,
        "features": features,
        "labels": labels,
        "initial_labeled_count": len(labeled_indices),
        "initial_unlabeled_count": len(unlabeled_indices),
        "test_count": len(test_indices),
        "random": random_history,
        "uncertainty": uncertainty_history,
        "curve_data": curve_data,
    }


def build_curve_data(
    random_history: list[dict],
    uncertainty_history: list[dict],
) -> dict:
    """Build a plot-ready curve data structure for report figures."""
    return {
        "random": {
            "round": [item["round"] for item in random_history],
            "labeled_samples": [item["labeled_samples"] for item in random_history],
            "accuracy": [item["accuracy"] for item in random_history],
            "precision": [item["precision"] for item in random_history],
            "recall": [item["recall"] for item in random_history],
            "f1": [item["f1"] for item in random_history],
        },
        "uncertainty": {
            "round": [item["round"] for item in uncertainty_history],
            "labeled_samples": [
                item["labeled_samples"] for item in uncertainty_history
            ],
            "accuracy": [item["accuracy"] for item in uncertainty_history],
            "precision": [item["precision"] for item in uncertainty_history],
            "recall": [item["recall"] for item in uncertainty_history],
            "f1": [item["f1"] for item in uncertainty_history],
        },
    }


def print_strategy_history(history: list[dict]) -> None:
    """Print one strategy's active learning curve as a compact table."""
    print(f"strategy: {history[0]['strategy']}")
    print("round | labeled | accuracy | precision | recall | F1")

    for item in history:
        print(
            f"{item['round']:>5} | "
            f"{item['labeled_samples']:>7} | "
            f"{item['accuracy']:.4f} | "
            f"{item['precision']:.4f} | "
            f"{item['recall']:.4f} | "
            f"{item['f1']:.4f}"
        )


def print_round_by_round_comparison(result: dict) -> None:
    """Print random-vs-uncertainty metrics in one side-by-side table."""
    print("round-by-round comparison")
    print(
        "round | labeled | "
        "random_acc | uncertain_acc | "
        "random_f1 | uncertain_f1 | "
        "random_precision | uncertain_precision | "
        "random_recall | uncertain_recall"
    )

    for random_item, uncertainty_item in zip(result["random"], result["uncertainty"]):
        print(
            f"{random_item['round']:>5} | "
            f"{random_item['labeled_samples']:>7} | "
            f"{random_item['accuracy']:.4f} | "
            f"{uncertainty_item['accuracy']:.4f} | "
            f"{random_item['f1']:.4f} | "
            f"{uncertainty_item['f1']:.4f} | "
            f"{random_item['precision']:.4f} | "
            f"{uncertainty_item['precision']:.4f} | "
            f"{random_item['recall']:.4f} | "
            f"{uncertainty_item['recall']:.4f}"
        )


def _save_metric_curve(
    curve_data: dict,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    """Save one random-vs-uncertainty metric curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        curve_data["random"]["labeled_samples"],
        curve_data["random"][metric],
        marker="o",
        label="Random sampling",
    )
    plt.plot(
        curve_data["uncertainty"]["labeled_samples"],
        curve_data["uncertainty"][metric],
        marker="o",
        label="Uncertainty sampling",
    )
    plt.xlabel("Labeled samples")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def save_active_learning_plots(curve_data: dict) -> list[Path]:
    """Save separate accuracy and F1 comparison curves for the report."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is not available in the current environment")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    return [
        _save_metric_curve(
            curve_data,
            metric="accuracy",
            ylabel="Accuracy",
            title="Active Learning Accuracy Comparison, s01",
            output_path=ACTIVE_LEARNING_ACCURACY_FIGURE_PATH,
        ),
        _save_metric_curve(
            curve_data,
            metric="f1",
            ylabel="F1 score",
            title="Active Learning F1 Comparison, s01",
            output_path=ACTIVE_LEARNING_F1_FIGURE_PATH,
        ),
    ]


def print_active_learning_report(result: dict) -> None:
    """Print the random-vs-uncertainty active learning comparison."""
    labels = result["labels"]
    positive_count = int(np.sum(labels == POSITIVE_LABEL))
    negative_count = int(np.sum(labels == NEGATIVE_LABEL))

    print("DEAP single-subject active learning experiment")
    print("note: this is a single-subject, small-sample, course-demo version.")
    print(f"subject: s{result['subject_id']:02d}")
    print(f"X shape: {result['features'].shape}")
    print(f"y shape: {labels.shape}")
    print(f"positive samples: {positive_count}")
    print(f"negative samples: {negative_count}")
    print(f"initial labeled ratio: {INITIAL_LABELED_RATIO:.0%}")
    print(f"initial labeled samples: {result['initial_labeled_count']}")
    print(f"unlabeled pool samples: {result['initial_unlabeled_count']}")
    print(f"fixed test samples: {result['test_count']}")
    print(f"query size per round: {QUERY_SIZE}")
    print()
    print_strategy_history(result["random"])
    print()
    print_strategy_history(result["uncertainty"])
    print()
    print_round_by_round_comparison(result)
    print()
    print("plot-ready performance curve data:")
    print("curve_data keys: random, uncertainty")
    print("each strategy contains: round, labeled_samples, accuracy, precision, recall, f1")
    print(
        "experiment note: this is a single-subject, small-sample, "
        "course-assignment minimum validation experiment."
    )

    try:
        figure_paths = save_active_learning_plots(result["curve_data"])
        for figure_path in figure_paths:
            print(f"saved figure: {figure_path}")
    except Exception as exc:
        print(f"figures were not saved because plotting failed: {exc}")


def main() -> None:
    """Run the minimal active learning comparison."""
    set_random_seed(RANDOM_STATE)
    result = run_active_learning_experiment()
    print_active_learning_report(result)


if __name__ == "__main__":
    main()
