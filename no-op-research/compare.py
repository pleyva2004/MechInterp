"""Check that the two stacks agree.   Run after both scripts:  uv run python compare.py"""
import numpy as np

from common import RESULTS, classify

hf, tl = (np.load(RESULTS / f"exp1_{name}.npz")["attn"] for name in ("hf", "tl"))
print("attention array shapes (hf, tl):", hf.shape, tl.shape)
print("largest attention difference:", np.abs(hf - tl).max())
print(f"per-sequence head classifications that agree: {100 * (classify(hf) == classify(tl)).mean():.2f}%")
