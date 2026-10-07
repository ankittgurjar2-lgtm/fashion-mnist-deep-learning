import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

FIG_DIR = "figures"

layers = [
    ("Input\n28x28x1", "#E8E8E8"),
    ("Conv 3x3, 32\n+ BatchNorm + ReLU", "#AEC7E8"),
    ("Conv 3x3, 32\n+ BatchNorm + ReLU", "#AEC7E8"),
    ("MaxPool 2x2\n+ Dropout 0.25\n-> 14x14x32", "#C7E9C0"),
    ("Conv 3x3, 64\n+ BatchNorm + ReLU", "#AEC7E8"),
    ("Conv 3x3, 64\n+ BatchNorm + ReLU", "#AEC7E8"),
    ("MaxPool 2x2\n+ Dropout 0.25\n-> 7x7x64", "#C7E9C0"),
    ("Flatten\n-> 3136", "#F7E9A0"),
    ("Dense 256\n+ ReLU + Dropout 0.5", "#F4B183"),
    ("Dense 10\n(logits)", "#F4B183"),
    ("Softmax\n-> class probabilities", "#E8E8E8"),
]

fig, ax = plt.subplots(figsize=(4.6, 11))
box_w, box_h, gap = 3.2, 0.78, 0.28
x0 = 0.4
y = len(layers) * (box_h + gap)
positions = []

for label, color in layers:
    y -= (box_h + gap)
    positions.append(y)
    box = FancyBboxPatch((x0, y), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.1, edgecolor="#444444", facecolor=color)
    ax.add_patch(box)
    ax.text(x0 + box_w / 2, y + box_h / 2, label, ha="center", va="center", fontsize=9.5)

for i in range(len(layers) - 1):
    y_top = positions[i]
    y_bot = positions[i + 1] + box_h
    ax.add_patch(FancyArrowPatch((x0 + box_w / 2, y_top), (x0 + box_w / 2, y_bot),
                                  arrowstyle="-|>", mutation_scale=14, color="#444444", linewidth=1.1))

ax.set_xlim(0, x0 * 2 + box_w)
ax.set_ylim(positions[-1] - 0.3, positions[0] + box_h + 0.6)
ax.axis("off")
ax.set_title("Final CNN architecture\n(871,018 trainable parameters)", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/00_architecture_diagram.png", dpi=150)
plt.close()
print("Saved architecture diagram")
