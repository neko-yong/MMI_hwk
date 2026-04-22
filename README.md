# EEG Emotion Recognition Course Project

## Project Goal

This project is a Human-Computer Interaction course assignment focused on emotion recognition based on EEG signals.

The current goal is to build a minimal, reproducible baseline with the following scope:

- Use a pure Python workflow for EEG preprocessing, feature extraction, and emotion classification.
- Prefer the DEAP dataset as the primary experimental dataset.
- First complete a minimal binary classification baseline:
  - task: positive vs negative emotion
  - model: SVM
- Then extend the project with an active learning setting and compare:
  - random sampling
  - active learning sampling
- Output standard experimental results, including:
  - accuracy
  - precision
  - recall
  - F1 score
  - confusion matrix

At this stage, the repository only contains the minimum project structure. The implementation is intentionally simple so it can be expanded step by step during the course project.

## Course Assignment Tasks

This project is intended to complete the following four parts of the course assignment:

1. EEG preprocessing
2. Emotion classification modeling
3. Active learning
4. Result analysis and discussion of human-computer interaction application prospects

## Dataset

- Recommended dataset: DEAP
- Dataset location: `data/DEAP/`

## Label Definition

This project uses valence scores from DEAP for binary classification:

- `positive`: `valence > 5`
- `negative`: `valence <= 5`

## Project Structure

```text
MMI_hwk/
├── data/
│   └── DEAP/
├── requirements.txt
├── README.md
├── results/
└── src/
    ├── __init__.py
    ├── preprocess.py
    ├── features.py
    ├── train_baseline.py
    ├── active_learning.py
    └── utils.py
```

## Module Responsibilities

- `preprocess.py`: EEG data preprocessing, such as channel selection, filtering, normalization, and basic signal cleaning.
- `features.py`: feature extraction from preprocessed EEG signals for later classification.
- `train_baseline.py`: baseline experiment entry, mainly for binary emotion classification with SVM.
- `active_learning.py`: active learning experiment entry, used to compare random sampling and active learning strategies.
- `utils.py`: shared helper functions, such as random seed control, path handling, and common utilities.

## Output Directory

- `results/`: stores experiment outputs such as evaluation metrics, plots, and confusion matrices.

## Environment Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Current Development Plan

The next implementation steps are:

1. Add DEAP data loading logic.
2. Implement basic EEG preprocessing.
3. Implement simple handcrafted features.
4. Train an SVM baseline for binary classification.
5. Add active learning and compare with random sampling.
6. Save evaluation metrics and figures for presentation.

## How To Run

The current version is only a scaffold, so the scripts mainly serve as placeholders for the next stage of development.

Before running, place the DEAP dataset under:

```text
data/DEAP/
```

You can test that the project structure is ready with:

```bash
python -m src.train_baseline
python -m src.active_learning
```

These commands currently print placeholder messages and do not yet run a full DEAP experiment.

## Notes

- The codebase is designed to stay modular and suitable for course presentation.
- The first implementation target is a minimal runnable baseline rather than a complex pipeline.
- More advanced preprocessing and modeling steps will be added incrementally.
