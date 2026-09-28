"""Shared pieces of Experiment 1: the random tokens and the analysis.

exp1_hf.py and exp1_tl.py differ only in how they pull attention out of the model.
Everything else lives here so the comparison between the two stacks stays fair.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

N = 100  # number of random sequences
SEQ_LEN = 32  # tokens per sequence
SEED = 0
CLASSES = ["first token", "previous token", "other token"]
COLORS = ["tab:red", "tab:blue", "lightgray"]
RESULTS = Path(__file__).parent / "results"


def make_tokens(vocab_size: int) -> torch.Tensor:
    """N sequences of SEQ_LEN uniformly random token ids, shape [N, SEQ_LEN].

    Same seed -> same tokens in both backends. Raw ids are fed straight to the model,
    so no BOS token is added.
    """
    gen = torch.Generator().manual_seed(SEED)
    return torch.randint(0, vocab_size, (N, SEQ_LEN), generator=gen)


def classify(attn: np.ndarray) -> np.ndarray:
    """attn: [N, layer, head, key] = attention the final position pays to each token.

    Returns [N, layer, head]: 0 = first token, 1 = previous token, 2 = other token,
    according to which token gets the most attention.
    """
    top = attn.argmax(-1)
    last = attn.shape[-1] - 1  # index of the final token itself
    cls = np.full(top.shape, 2)
    cls[top == 0] = 0
    cls[top == last - 1] = 1
    return cls


def analyze(attn: np.ndarray, name: str) -> None:
    """Print the summary, then save the raw attention and the figure to results/."""
    cls = classify(attn)
    share = np.stack([(cls == c).mean(axis=0) for c in range(3)])  # [3, layer, head]
    head_class = share.argmax(axis=0)  # each head's most common target (ties go to first, then previous)

    for c, label in enumerate(CLASSES):
        print(f"[{name}] {100 * (head_class == c).mean():.1f}% of heads primarily attend to the {label}")
    print(f"[{name}] layer x head  (F = first, P = previous, O = other)")
    for layer, row in enumerate(head_class):
        print(f"  L{layer:<2} " + " ".join("FPO"[c] for c in row))

    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / f"exp1_{name}.npz", attn=attn)  # raw final-row attention: re-classify later without re-running
    plot(attn[..., 0].mean(axis=0), head_class, RESULTS / f"exp1_{name}.png")


def plot(mean_first: np.ndarray, head_class: np.ndarray, path: Path) -> None:
    """Two layer x head heatmaps: attention on the first token, and each head's main target."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    im = left.imshow(mean_first, origin="lower", vmin=0, vmax=1)
    left.set_title("Mean attention on the first token")
    fig.colorbar(im, ax=left)

    right.imshow(head_class, origin="lower", cmap=ListedColormap(COLORS), vmin=0, vmax=2)
    right.set_title("What each head attends to most")
    right.legend(
        handles=[Patch(color=c, label=l) for c, l in zip(COLORS, CLASSES)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )

    for ax in (left, right):
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
