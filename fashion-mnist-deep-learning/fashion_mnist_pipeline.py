"""
Week 5 Task — Deep Learning Application in Data Science
Dataset : Fashion-MNIST (60,000 train / 10,000 test, 28x28 grayscale, 10 classes)
Source  : Zalando Research, mirrored at https://github.com/zalandoresearch/fashion-mnist
Problem : Multi-class image classification — identify the garment category from a
          28x28 grayscale product photo.

This script:
  1. Loads Fashion-MNIST from the raw IDX files (no internet calls needed after
     the one-time GitHub clone used to fetch them).
  2. Builds three models for a deliberate, informative comparison:
       - MLP baseline (no convolutional structure at all)
       - CNN without regularization (to deliberately demonstrate overfitting)
       - CNN with BatchNorm + Dropout (the final, regularized model)
  3. Trains all three with identical data, optimizer, and epoch budget.
  4. Evaluates the final model on the untouched 10,000-image test set:
     accuracy, per-class precision/recall/F1, confusion matrix.
  5. Saves learning curves, a confusion matrix, sample images, misclassified
     examples, and an architecture diagram.

Run: python3 fashion_mnist_pipeline.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

torch.manual_seed(42)
np.random.seed(42)
sns.set_theme(style="whitegrid", font_scale=0.95)
FIG_DIR = "figures"
DEVICE = torch.device("cpu")
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ---------------------------------------------------------------------------
# 0. LOAD DATA
# ---------------------------------------------------------------------------
def read_idx_images(path):
    with open(path, "rb") as f:
        f.read(4)
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    return data


def read_idx_labels(path):
    with open(path, "rb") as f:
        f.read(4)
        n = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


section("0. LOADING FASHION-MNIST")
X_train_full = read_idx_images("data/train-images-idx3-ubyte")
y_train_full = read_idx_labels("data/train-labels-idx1-ubyte")
X_test = read_idx_images("data/t10k-images-idx3-ubyte")
y_test = read_idx_labels("data/t10k-labels-idx1-ubyte")
print(f"Train pool: {X_train_full.shape}, Test: {X_test.shape}")
print(f"Classes: {CLASSES}")


class FashionMNIST(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].astype(np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()  # random horizontal flip (valid for clothing)
        img = (img - 0.2860) / 0.3530  # Fashion-MNIST global mean/std
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return tensor, int(self.y[idx])


# NOTE ON SCALE: this machine trains on a single CPU core (no GPU). A quick timing
# test showed ~86s/epoch on the full 50,000-image training split for the CNN below,
# which would make a 3-model x multi-epoch comparison take the better part of an hour.
# Rather than silently cutting epochs to the point the learning curves stop being
# informative, an 8,000/2,000 train/validation subset of the 60,000-image training
# pool is used instead, keeping the full, untouched 10,000-image test set for final
# evaluation. This is itself one of the CPU-resource-constraint decisions discussed
# in the report.
rng = np.random.RandomState(42)
perm = rng.permutation(len(X_train_full))
train_idx, val_idx = perm[:8000], perm[8000:10000]

train_ds = FashionMNIST(X_train_full[train_idx], y_train_full[train_idx], augment=True)
val_ds = FashionMNIST(X_train_full[val_idx], y_train_full[val_idx], augment=False)
test_ds = FashionMNIST(X_test, y_test, augment=False)

BATCH_SIZE = 256
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# ---------------------------------------------------------------------------
# 1. MODEL DEFINITIONS
# ---------------------------------------------------------------------------
section("1. MODEL DEFINITIONS")


class MLP(nn.Module):
    """Fully-connected baseline — no spatial structure at all."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Small CNN. regularize=False reproduces an overfitting-prone version
    (no BatchNorm, no Dropout) for Section 3's ablation; regularize=True is
    the final model used for test-set evaluation."""
    def __init__(self, regularize=True):
        super().__init__()
        self.regularize = regularize
        p_conv, p_fc = (0.25, 0.5) if regularize else (0.0, 0.0)

        def bn(c):
            return nn.BatchNorm2d(c) if regularize else nn.Identity()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), bn(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), bn(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(p_conv),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), bn(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), bn(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(p_conv),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(p_fc),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


for name, m in [("MLP", MLP()), ("CNN", CNN())]:
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{name}: {n_params:,} parameters")

# ---------------------------------------------------------------------------
# 2. TRAINING LOOP (shared by all three models)
# ---------------------------------------------------------------------------
section("2. TRAINING")


def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if train_mode:
            optimizer.zero_grad()
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n


def train_model(model, name, epochs):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, val_loader, optimizer=None)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"[{name}] epoch {epoch:2d}/{epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
              f"({time.time()-t0:.1f}s elapsed)")
    return history


EPOCHS = 8

print("\n--- Training MLP baseline ---")
mlp = MLP()
hist_mlp = train_model(mlp, "MLP", EPOCHS)

print("\n--- Training CNN WITHOUT regularization (overfitting demo) ---")
cnn_noreg = CNN(regularize=False)
hist_cnn_noreg = train_model(cnn_noreg, "CNN-no-reg", EPOCHS)

print("\n--- Training CNN WITH BatchNorm + Dropout (final model) ---")
cnn_final = CNN(regularize=True)
hist_cnn_final = train_model(cnn_final, "CNN-final", EPOCHS)

# ---------------------------------------------------------------------------
# 3. LEARNING CURVES — OVERFITTING COMPARISON
# ---------------------------------------------------------------------------
section("3. LEARNING CURVES")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for h, label, color in [(hist_cnn_noreg, "CNN, no regularization", "#C44E52"),
                         (hist_cnn_final, "CNN, BatchNorm + Dropout", "#4C72B0")]:
    epochs_range = range(1, EPOCHS + 1)
    axes[0].plot(epochs_range, h["train_acc"], color=color, ls="--", alpha=0.7)
    axes[0].plot(epochs_range, h["val_acc"], color=color, label=label)
axes[0].set_title("Train (dashed) vs. validation (solid) accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend(fontsize=8)

gap_noreg = [t - v for t, v in zip(hist_cnn_noreg["train_acc"], hist_cnn_noreg["val_acc"])]
gap_final = [t - v for t, v in zip(hist_cnn_final["train_acc"], hist_cnn_final["val_acc"])]
axes[1].plot(range(1, EPOCHS + 1), gap_noreg, color="#C44E52", label="No regularization")
axes[1].plot(range(1, EPOCHS + 1), gap_final, color="#4C72B0", label="BatchNorm + Dropout")
axes[1].set_title("Train-minus-validation accuracy gap\n(larger = more overfitting)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Train acc. \u2212 Val acc.")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_overfitting_comparison.png", dpi=150)
plt.close()

print(f"Final-epoch train-val gap — no regularization: {gap_noreg[-1]:.4f}, "
      f"with regularization: {gap_final[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for h, label, color in [(hist_mlp, "MLP baseline", "#55A868"),
                         (hist_cnn_final, "CNN (final)", "#4C72B0")]:
    axes[0].plot(range(1, EPOCHS + 1), h["val_acc"], color=color, label=label)
    axes[1].plot(range(1, EPOCHS + 1), h["val_loss"], color=color, label=label)
axes[0].set_title("Validation accuracy: MLP vs. CNN")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation accuracy")
axes[0].legend(fontsize=8)
axes[1].set_title("Validation loss: MLP vs. CNN")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation loss")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_mlp_vs_cnn.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. FINAL TEST-SET EVALUATION (CNN-final only, touched exactly once)
# ---------------------------------------------------------------------------
section("4. FINAL TEST-SET EVALUATION (CNN with BatchNorm + Dropout)")

cnn_final.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = cnn_final(xb.to(DEVICE))
        all_preds.append(out.argmax(1).numpy())
        all_true.append(yb.numpy())
y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_true)

test_acc = accuracy_score(y_true, y_pred)
print(f"Test accuracy: {test_acc:.4f}")
print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=CLASSES))

# MLP test accuracy for comparison
mlp.eval()
mlp_preds = []
with torch.no_grad():
    for xb, yb in test_loader:
        out = mlp(xb.to(DEVICE))
        mlp_preds.append(out.argmax(1).numpy())
mlp_preds = np.concatenate(mlp_preds)
mlp_test_acc = accuracy_score(y_true, mlp_preds)
print(f"\nMLP baseline test accuracy: {mlp_test_acc:.4f}")
print(f"CNN (no reg) would-be test accuracy for reference:")
cnn_noreg.eval()
noreg_preds = []
with torch.no_grad():
    for xb, yb in test_loader:
        out = cnn_noreg(xb.to(DEVICE))
        noreg_preds.append(out.argmax(1).numpy())
noreg_preds = np.concatenate(noreg_preds)
print(f"  {accuracy_score(y_true, noreg_preds):.4f}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7.5, 6.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f"Confusion matrix — final CNN (test accuracy = {test_acc:.3f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. SAMPLE IMAGES AND MISCLASSIFIED EXAMPLES
# ---------------------------------------------------------------------------
section("5. SAMPLE IMAGES")

fig, axes = plt.subplots(2, 5, figsize=(10, 5.2))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train_full == i)[0][0]
    ax.imshow(X_train_full[idx], cmap="gray")
    ax.set_title(CLASSES[i], fontsize=9)
    ax.axis("off")
plt.suptitle("One example of each Fashion-MNIST class")
plt.subplots_adjust(hspace=0.4, top=0.88)
plt.savefig(f"{FIG_DIR}/01_sample_images.png", dpi=150)
plt.close()

wrong_idx = np.where(y_pred != y_true)[0]
print(f"Misclassified test images: {len(wrong_idx)} / {len(y_true)} ({len(wrong_idx)/len(y_true):.1%})")
rng = np.random.RandomState(42)
chosen = rng.choice(wrong_idx, size=10, replace=False)
fig, axes = plt.subplots(2, 5, figsize=(10, 5.8))
for ax, idx in zip(axes.flat, chosen):
    ax.imshow(X_test[idx], cmap="gray")
    ax.set_title(f"true: {CLASSES[y_true[idx]]}\npred: {CLASSES[y_pred[idx]]}", fontsize=8)
    ax.axis("off")
plt.suptitle("10 randomly sampled misclassified test images")
plt.subplots_adjust(hspace=0.5, top=0.88)
plt.savefig(f"{FIG_DIR}/05_misclassified.png", dpi=150)
plt.close()

# Which class pairs get confused most (off-diagonal hot spots)
cm_offdiag = cm.copy()
np.fill_diagonal(cm_offdiag, 0)
top_confusions = np.dstack(np.unravel_index(np.argsort(-cm_offdiag.ravel())[:5], cm_offdiag.shape))[0]
print("\nTop 5 confused class pairs (true -> predicted, count):")
for t, p in top_confusions:
    print(f"  {CLASSES[t]:15s} -> {CLASSES[p]:15s} : {cm[t, p]}")

torch.save(cnn_final.state_dict(), "models/cnn_final.pt")
np.savez("data/test_predictions.npz", y_true=y_true, y_pred=y_pred, mlp_pred=mlp_preds, noreg_pred=noreg_preds)
pd.DataFrame({"epoch": range(1, EPOCHS + 1), **{f"mlp_{k}": v for k, v in hist_mlp.items()},
              **{f"cnn_noreg_{k}": v for k, v in hist_cnn_noreg.items()},
              **{f"cnn_final_{k}": v for k, v in hist_cnn_final.items()}}).to_csv("data/training_history.csv", index=False)

print("\nAll figures written to ./figures/")
print("DONE.")
