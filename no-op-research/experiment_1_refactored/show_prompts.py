"""Show what the Experiment-1 prompts actually look like: 32 independently random
token ids per sequence, decoded back to text. Demonstrates that they're gibberish,
not sentences.

Every box is sized to its own token's rendered text, measured with matplotlib's
renderer, so nothing is ever clipped or overlapping regardless of token length.
Sequences wrap onto as many lines as their tokens need.

Run:  uv run python show_prompts.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoTokenizer

from common import RESULTS, SEQ_LEN, make_tokens

N_EXAMPLES = 4  # how many example sequences to draw
PIECE_FONT, ID_FONT = 11, 7
PAD_X, GAP_X = 0.14, 0.08  # inches: padding inside a box, gap between boxes
MIN_BOX_W, BOX_H = 0.55, 0.8  # inches
ROW_WIDTH = 15.5  # inches: wrap a sequence onto a new line past this width
ROW_GAP, SEQ_GAP = 0.12, 0.35  # inches: gap between wrapped lines / between sequences
LEFT_MARGIN, RIGHT_MARGIN, TOP_MARGIN, BOTTOM_MARGIN = 0.9, 0.3, 0.6, 0.2

tok = AutoTokenizer.from_pretrained("gpt2")
tokens = make_tokens(tok.vocab_size)  # same seeded tokens exp1_hf.py / exp1_tl.py use

sequences = []
for row in range(N_EXAMPLES):
    ids = tokens[row].tolist()
    pieces = [tok.decode([t]).strip() or "·" for t in ids]
    sequences.append(list(zip(pieces, ids)))

# --- Measure every box's exact width first, on a throwaway figure/renderer. ---
meas_fig, meas_ax = plt.subplots()
meas_fig.canvas.draw()
renderer = meas_fig.canvas.get_renderer()


def text_width_in(text: str, fontsize: int) -> float:
    artist = meas_ax.text(0, 0, text, fontsize=fontsize)
    width = artist.get_window_extent(renderer=renderer).transformed(meas_fig.dpi_scale_trans.inverted()).width
    artist.remove()
    return width


boxes_by_seq = []  # per sequence: list of (piece, id_str, box_width)
for seq in sequences:
    boxes = []
    for piece, tid in seq:
        id_str = str(tid)
        w = max(text_width_in(piece, PIECE_FONT), text_width_in(id_str, ID_FONT)) + 2 * PAD_X
        boxes.append((piece, id_str, max(w, MIN_BOX_W)))
    boxes_by_seq.append(boxes)
plt.close(meas_fig)

# --- Flow-wrap each sequence's boxes into as few lines as fit within ROW_WIDTH. ---
layout_by_seq = []  # per sequence: list of lines, each a list of (piece, id_str, x, box_w)
for boxes in boxes_by_seq:
    lines, line, x = [], [], 0.0
    for piece, id_str, w in boxes:
        if line and x + w > ROW_WIDTH:
            lines.append(line)
            line, x = [], 0.0
        line.append((piece, id_str, x, w))
        x += w + GAP_X
    lines.append(line)
    layout_by_seq.append(lines)

n_lines = [len(lines) for lines in layout_by_seq]
total_h = sum(n_lines) * BOX_H + (sum(n_lines) - N_EXAMPLES) * ROW_GAP + (N_EXAMPLES - 1) * SEQ_GAP
fig_w = LEFT_MARGIN + ROW_WIDTH + RIGHT_MARGIN
fig_h = TOP_MARGIN + total_h + BOTTOM_MARGIN

fig = plt.figure(figsize=(fig_w, fig_h))
ax = fig.add_axes((0, 0, 1, 1))  # fill the whole figure: 1 data unit == 1 inch, matching the margins above exactly
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")
ax.text(
    fig_w / 2, fig_h - TOP_MARGIN * 0.5,
    f"Experiment 1 prompts: {SEQ_LEN} independently random token ids, decoded to text",
    ha="center", va="center", fontsize=14,
)

y = fig_h - TOP_MARGIN
for seq_idx, lines in enumerate(layout_by_seq):
    seq_top, seq_bottom = y, y - len(lines) * BOX_H - (len(lines) - 1) * ROW_GAP
    ax.text(
        LEFT_MARGIN - 0.15, (seq_top + seq_bottom) / 2, f"seq {seq_idx}",
        ha="right", va="center", fontsize=9, color="dimgray",
    )
    for line in lines:
        for piece, id_str, x, w in line:
            box_x = LEFT_MARGIN + x
            ax.add_patch(Rectangle((box_x, y - BOX_H), w, BOX_H, fill=False, edgecolor="tab:blue", linewidth=0.9))
            ax.text(box_x + w / 2, y - 0.32 * BOX_H, piece, ha="center", va="center", fontsize=PIECE_FONT)
            ax.text(box_x + w / 2, y - 0.75 * BOX_H, id_str, ha="center", va="center", fontsize=ID_FONT, color="gray")
        y -= BOX_H + ROW_GAP
    y = seq_bottom - SEQ_GAP + ROW_GAP  # undo the extra ROW_GAP subtracted after the last line

path = RESULTS / "exp1_prompts.png"
RESULTS.mkdir(exist_ok=True)
fig.savefig(path, dpi=150, bbox_inches="tight")
print(f"saved {path}")
