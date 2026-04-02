# AI Generated Image Detection

This project restructures the provided challenge assets into a small PyTorch project with:

- a reusable `src/ai_image_detection` package
- a training notebook in `notebooks/`
- clean helpers for loading data, training a CNN, evaluating results, and creating submissions

The workflow follows the same shape as the FashionMNIST tutorial:

1. inspect and visualize the dataset
2. build `Dataset` and `DataLoader` objects
3. define a baseline computer vision model
4. train with reusable train/eval loops
5. evaluate with metrics and a confusion matrix
6. save the model and create test predictions

## Project Structure

```text
.
├── notebooks/
│   └── 01_ai_generated_image_detection.ipynb
├── src/
│   └── ai_image_detection/
│       ├── __init__.py
│       ├── data.py
│       ├── engine.py
│       └── models.py
├── requirements.txt
└── [External] DCU 2026 ML challenge - external/
    ├── train.csv
    ├── test.csv
    └── images_final_sample/
```

## Dataset Notes

The current repository snapshot includes:

- `train.csv` with `4800` labels
- `test.csv` with `2058` unlabeled rows
- `images_final_sample/` with `6200` image files

That means not every ID from the CSV files is available locally right now. The notebook and data helpers automatically filter to rows whose image files actually exist, so you can still train and evaluate on the available subset.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open:

`notebooks/01_ai_generated_image_detection.ipynb`

## Output

The notebook is set up to generate:

- a trained CNN checkpoint
- validation metrics
- a confusion matrix
- a `submission.csv` for the available test images
