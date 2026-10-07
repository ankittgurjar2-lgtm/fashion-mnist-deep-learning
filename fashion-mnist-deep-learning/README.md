# Fashion-MNIST — Deep Learning Image Classification

Week 5 task for a data science internship: design, train, and evaluate a neural
network. This week deliberately switches dataset and framework from the
Titanic-based Weeks 1-4 of this internship, since deep
learning is better demonstrated on high-dimensional image data than on a 6-feature
tabular problem a shallow model already solves well.

## Problem

Multi-class image classification: identify the garment category (10 classes) from a
28x28 grayscale Fashion-MNIST product photo.

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research):
60,000 training + 10,000 test grayscale images, 10 balanced classes. The raw IDX files
are not included in this repo (kept small on purpose) — fetch them with:

```bash
git clone --depth 1 https://github.com/zalandoresearch/fashion-mnist.git
cp fashion-mnist/data/fashion/*.gz data/ && cd data && gunzip -k *.gz
```

## What the pipeline does

`fashion_mnist_pipeline.py` (PyTorch):

1. **Data loading** — parses the raw IDX files directly (no torchvision downloader,
   since its default URLs aren't reachable from this environment)
2. **Three models, trained identically**, for a controlled comparison:
   - MLP baseline (235K params, no convolutional structure)
   - CNN without BatchNorm/Dropout (871K params — isolates the regularization effect)
   - **Final CNN**: BatchNorm + Dropout (871K params) — see `draw_architecture.py` for
     the architecture diagram
3. **Evaluation** — learning curves, an overfitting (train-vs-val gap) comparison,
   final test-set accuracy/precision/recall/F1, confusion matrix, and a misclassified-
   example gallery

Full narrative, code, the architecture diagram, and all 6 result figures are written
up in [`Week5_DeepLearning_Report.docx`](./Week5_DeepLearning_Report.docx).

## Results

| Model | Test accuracy |
|---|---|
| MLP baseline | 83.96% |
| CNN, no regularization | 86.08% |
| **CNN, BatchNorm + Dropout (final)** | **87.98%** |

Weakest class: **Shirt** (0.68 F1) — confused mainly with T-shirt/top, Pullover, and
Coat, which the report traces to genuine visual ambiguity at 28x28 resolution rather
than a fixable modeling issue (Section 9 of the report).

**Resource-constraint note:** this was trained on a single CPU core (no GPU), so
training/validation used an 8,000/2,000-image subset of the 60,000-image training pool
rather than the full set, to keep an 8-epoch, 3-model comparison under ~10 minutes.
The full, official 10,000-image test set was used untouched for all reported results.

## Running it

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn
python draw_architecture.py          # architecture diagram only, no data needed
python fashion_mnist_pipeline.py     # full training + evaluation (~8-10 min on CPU)
```

## Structure

```
fashion_mnist_pipeline.py        # full training/evaluation pipeline
draw_architecture.py             # generates the architecture diagram figure
models/cnn_final.pt              # trained weights for the final CNN
data/training_history.csv        # per-epoch loss/accuracy for all 3 models
figures/                         # 6 generated figures (architecture, curves, confusion matrix, etc.)
Week5_DeepLearning_Report.docx   # full written report
```
