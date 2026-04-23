"""Minimal SVM baseline for single-subject DEAP valence classification.

This script is intentionally small and course-demo oriented. It uses one
subject's 40 trials only, so the result is a pipeline sanity check rather than
a strong experimental conclusion. The next step can extend this baseline to
multiple subjects.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.features import extract_features
from src.preprocess import (
    DEFAULT_METADATA_DIR,
    DEFAULT_ORIGINAL_DIR,
    PROJECT_ROOT,
    extract_raw_eeg_trials_from_bdf,
    load_participant_ratings,
    run_basic_preprocessing_on_standardized_trials,
    standardize_raw_eeg_trial_lengths,
)
from src.utils import set_random_seed


POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
VALENCE_THRESHOLD = 5.0
DEFAULT_SUBJECT_ID = 1
RANDOM_STATE = 42
TEST_SIZE = 0.25
SVM_EPOCHS = 3000
SVM_LEARNING_RATE = 0.001
SVM_REGULARIZATION = 0.01
RESULTS_DIR = Path("results")
CONFUSION_MATRIX_FIGURE_PATH = RESULTS_DIR / "baseline_s01_confusion_matrix.png"
CLASS_DISTRIBUTION_FIGURE_PATH = RESULTS_DIR / "baseline_s01_class_distribution.png"


try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_valence_binary_labels(subject_id: int = DEFAULT_SUBJECT_ID) -> np.ndarray:
    """Load one subject's DEAP valence labels and convert them to 0/1 labels.

    Label rule:
    - positive: valence > 5
    - negative: valence <= 5
    """
    ratings = load_participant_ratings(subject_id, DEFAULT_METADATA_DIR)
    labels = [
        POSITIVE_LABEL
        if float(row["Valence"]) > VALENCE_THRESHOLD
        else NEGATIVE_LABEL
        for row in ratings
    ]
    return np.asarray(labels, dtype=np.int64)


def build_single_subject_feature_matrix(
    subject_id: int = DEFAULT_SUBJECT_ID,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X/y for one DEAP subject from raw BDF and participant ratings."""
    bdf_path = PROJECT_ROOT / DEFAULT_ORIGINAL_DIR / f"s{subject_id:02d}.bdf"

    extracted = extract_raw_eeg_trials_from_bdf(bdf_path)
    standardized = standardize_raw_eeg_trial_lengths(extracted)
    preprocessed = run_basic_preprocessing_on_standardized_trials(standardized)

    stimulus_data = preprocessed["baseline_corrected_stimulus"]
    features = extract_features(
        stimulus_data,
        sampling_rate=int(preprocessed["sampling_rate"]),
    )
    labels = load_valence_binary_labels(subject_id)

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Feature/label count mismatch: "
            f"X has {features.shape[0]} trials, y has {labels.shape[0]} labels."
        )

    return features, labels


def train_and_evaluate_svm(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Train a minimal standardized SVM classifier and return metrics."""
    if SKLEARN_AVAILABLE:
        return _train_and_evaluate_sklearn_svm(features, labels, random_state)

    return _train_and_evaluate_numpy_linear_svm(features, labels, random_state)


def _train_and_evaluate_sklearn_svm(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> dict:
    """Train the preferred scikit-learn SVM baseline when sklearn exists."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=labels,
    )

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    return {
        "model": "sklearn.svm.SVC(kernel='rbf', C=1.0, gamma='scale')",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        ),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }


def _stratified_train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small stratified split helper used when scikit-learn is unavailable."""
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        label_test_count = max(1, int(round(len(label_indices) * test_size)))
        test_indices.extend(label_indices[:label_test_count])
        train_indices.extend(label_indices[label_test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        features[train_indices],
        features[test_indices],
        labels[train_indices],
        labels[test_indices],
    )


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize features using train-set statistics only."""
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def _fit_numpy_linear_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, float]:
    """Fit a tiny linear SVM with hinge loss using NumPy SGD.

    This fallback exists only to keep the course baseline runnable when
    scikit-learn is not installed. It is not a replacement for a full SVM
    library implementation.
    """
    rng = np.random.default_rng(random_state)
    y_signed = np.where(y_train == POSITIVE_LABEL, 1.0, -1.0)
    weights = np.zeros(x_train.shape[1], dtype=np.float64)
    bias = 0.0

    for _ in range(SVM_EPOCHS):
        for index in rng.permutation(len(y_signed)):
            x_i = x_train[index]
            y_i = y_signed[index]
            margin = y_i * (np.dot(weights, x_i) + bias)

            if margin >= 1.0:
                weights -= SVM_LEARNING_RATE * SVM_REGULARIZATION * weights
            else:
                weights -= SVM_LEARNING_RATE * (
                    SVM_REGULARIZATION * weights - y_i * x_i
                )
                bias += SVM_LEARNING_RATE * y_i

    return weights, bias


def _predict_numpy_linear_svm(
    x_test: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    """Predict labels with the NumPy linear SVM fallback."""
    scores = x_test @ weights + bias
    return np.where(scores >= 0, POSITIVE_LABEL, NEGATIVE_LABEL)


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute binary metrics without scikit-learn."""
    true_positive = int(np.sum((y_true == POSITIVE_LABEL) & (y_pred == POSITIVE_LABEL)))
    true_negative = int(np.sum((y_true == NEGATIVE_LABEL) & (y_pred == NEGATIVE_LABEL)))
    false_positive = int(np.sum((y_true == NEGATIVE_LABEL) & (y_pred == POSITIVE_LABEL)))
    false_negative = int(np.sum((y_true == POSITIVE_LABEL) & (y_pred == NEGATIVE_LABEL)))

    accuracy = (true_positive + true_negative) / len(y_true)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": np.asarray(
            [
                [true_negative, false_positive],
                [false_negative, true_positive],
            ],
            dtype=np.int64,
        ),
    }


def _train_and_evaluate_numpy_linear_svm(
    features: np.ndarray,
    labels: np.ndarray,
    random_state: int,
) -> dict:
    """Train a minimal NumPy linear SVM fallback and return metrics."""
    x_train, x_test, y_train, y_test = _stratified_train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=random_state,
    )
    x_train, x_test = _standardize_train_test(x_train, x_test)
    weights, bias = _fit_numpy_linear_svm(x_train, y_train, random_state)
    y_pred = _predict_numpy_linear_svm(x_test, weights, bias)
    metrics = _compute_binary_metrics(y_test, y_pred)

    return {
        "model": "NumPy fallback Linear SVM (hinge-loss SGD)",
        **metrics,
        "train_size": len(y_train),
        "test_size": len(y_test),
    }


def print_baseline_report(
    features: np.ndarray,
    labels: np.ndarray,
    metrics: dict,
    subject_id: int = DEFAULT_SUBJECT_ID,
) -> None:
    """Print shapes, label counts, and baseline classification metrics."""
    positive_count = int(np.sum(labels == POSITIVE_LABEL))
    negative_count = int(np.sum(labels == NEGATIVE_LABEL))

    print("DEAP single-subject SVM baseline")
    print(f"subject: s{subject_id:02d}")
    print(f"model: {metrics['model']}")
    print("note: this is a minimal 40-trial sanity-check baseline.")
    print("future work: extend the same pipeline to multiple subjects.")
    print(f"X shape: {features.shape}")
    print(f"y shape: {labels.shape}")
    print(f"positive samples: {positive_count}")
    print(f"negative samples: {negative_count}")
    print(f"train/test split: {metrics['train_size']}/{metrics['test_size']}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print("confusion matrix, labels=[negative, positive]:")
    print(metrics["confusion_matrix"])


def save_confusion_matrix_plot(
    confusion_matrix_values: np.ndarray,
    output_path: Path = CONFUSION_MATRIX_FIGURE_PATH,
) -> Path:
    """Save a confusion matrix figure for baseline result presentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion_matrix_values, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    class_names = ["negative", "positive"]
    axis.set_xticks(np.arange(len(class_names)), labels=class_names)
    axis.set_yticks(np.arange(len(class_names)), labels=class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("SVM Baseline Confusion Matrix, s01")

    for row_index in range(confusion_matrix_values.shape[0]):
        for col_index in range(confusion_matrix_values.shape[1]):
            axis.text(
                col_index,
                row_index,
                int(confusion_matrix_values[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_class_distribution_plot(
    labels: np.ndarray,
    output_path: Path = CLASS_DISTRIBUTION_FIGURE_PATH,
) -> Path:
    """Save a positive/negative sample-count bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    negative_count = int(np.sum(labels == NEGATIVE_LABEL))
    positive_count = int(np.sum(labels == POSITIVE_LABEL))

    fig, axis = plt.subplots(figsize=(5, 4))
    bars = axis.bar(
        ["negative", "positive"],
        [negative_count, positive_count],
        color=["#4C78A8", "#F58518"],
    )
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of samples")
    axis.set_title("Valence Label Distribution, s01")

    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_baseline_figures(labels: np.ndarray, metrics: dict) -> list[Path]:
    """Save baseline visualizations without changing experiment logic."""
    return [
        save_confusion_matrix_plot(metrics["confusion_matrix"]),
        save_class_distribution_plot(labels),
    ]


def main() -> None:
    """Run the minimal single-subject SVM baseline."""
    set_random_seed(42)
    subject_id = DEFAULT_SUBJECT_ID
    features, labels = build_single_subject_feature_matrix(subject_id)
    metrics = train_and_evaluate_svm(features, labels)
    print_baseline_report(features, labels, metrics, subject_id)
    figure_paths = save_baseline_figures(labels, metrics)
    for figure_path in figure_paths:
        print(f"saved figure: {figure_path}")


if __name__ == "__main__":
    main()
