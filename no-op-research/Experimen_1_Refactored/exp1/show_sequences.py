"""Render example input sequences as they were fed to the model, one box per token.

Ported from ../experiment_1_refactored/show_prompts.py. Ids are decoded to text for display
only; the model always receives the raw id tensor. Every box is sized to its own token's
rendered text, measured with matplotlib's renderer, so nothing is clipped or overlapping;
sequences wrap onto as many lines as their tokens need.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PIECE_FONT, ID_FONT = 11, 7
PAD_X, GAP_X = 0.14, 0.08  # inches: padding inside a box, gap between boxes
MIN_BOX_W, BOX_H = 0.55, 0.8  # inches
ROW_WIDTH = 15.5  # inches: wrap a sequence onto a new line past this width
ROW_GAP, SEQ_GAP = 0.12, 0.35  # inches: gap between wrapped lines / between sequences
LEFT_MARGIN, RIGHT_MARGIN, TOP_MARGIN, BOTTOM_MARGIN = 0.9, 0.3, 0.8, 0.2
TOKEN_EDGE, BOS_EDGE, BOS_FILL = "tab:blue", "dimgray", "0.9"


def _display_piece(tokenizer, token_id: int) -> tuple[str, bool]:
    """(text, is_symbol) for one id. Byte tokens (0-255) and whitespace often decode to nothing
    printable, so fall back to the byte-level BPE symbol (e.g. 'Ċ' for a newline) and flag it."""
    piece = tokenizer.decode([token_id]).strip()
    is_symbol = not piece or not piece.isprintable() or "�" in piece
    if is_symbol:
        piece = tokenizer.convert_ids_to_tokens(token_id)
    return piece.replace("$", r"\$"), is_symbol  # a pair of '$' would switch matplotlib into mathtext


def plot_sequences(
    tokens: np.ndarray, tokenizer, path: str | Path, title: str, n_examples: int, bos_id: int | None = None
) -> None:
    """tokens: int64 [n_seq, key_len], exactly what the model saw. Draws the first n_examples rows;
    a bos_id in column 0 gets its own gray box so it reads as the prepended sink token."""
    sequences = []
    for row in tokens[:n_examples]:
        seq = []
        for pos, tid in enumerate(row.tolist()):
            is_bos = pos == 0 and tid == bos_id
            piece, is_symbol = ("BOS", False) if is_bos else _display_piece(tokenizer, tid)
            seq.append((piece, f"{pos}:{tid}", is_bos, is_symbol))
        sequences.append(seq)

    # Measure every box's exact width first, on a throwaway figure/renderer.
    meas_fig, meas_ax = plt.subplots()
    meas_fig.canvas.draw()
    renderer = meas_fig.canvas.get_renderer()

    def text_width_in(text: str, fontsize: int) -> float:
        artist = meas_ax.text(0, 0, text, fontsize=fontsize)
        width = artist.get_window_extent(renderer=renderer).transformed(meas_fig.dpi_scale_trans.inverted()).width
        artist.remove()
        return width

    boxes_by_seq = [
        [(piece, id_str, is_bos, is_symbol,
          max(max(text_width_in(piece, PIECE_FONT), text_width_in(id_str, ID_FONT)) + 2 * PAD_X, MIN_BOX_W))
         for piece, id_str, is_bos, is_symbol in seq]
        for seq in sequences
    ]
    plt.close(meas_fig)

    # Flow-wrap each sequence's boxes into as few lines as fit within ROW_WIDTH.
    layout_by_seq = []  # per sequence: list of lines, each a list of (piece, id_str, is_bos, is_symbol, x, box_w)
    for boxes in boxes_by_seq:
        lines, line, x = [], [], 0.0
        for piece, id_str, is_bos, is_symbol, w in boxes:
            if line and x + w > ROW_WIDTH:
                lines.append(line)
                line, x = [], 0.0
            line.append((piece, id_str, is_bos, is_symbol, x, w))
            x += w + GAP_X
        lines.append(line)
        layout_by_seq.append(lines)

    n_seqs = len(layout_by_seq)
    n_lines = sum(len(lines) for lines in layout_by_seq)
    total_h = n_lines * BOX_H + (n_lines - n_seqs) * ROW_GAP + (n_seqs - 1) * SEQ_GAP
    fig_w = LEFT_MARGIN + ROW_WIDTH + RIGHT_MARGIN
    fig_h = TOP_MARGIN + total_h + BOTTOM_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((0, 0, 1, 1))  # fill the whole figure: 1 data unit == 1 inch, matching the margins above
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.text(fig_w / 2, fig_h - TOP_MARGIN * 0.35, title, ha="center", va="center", fontsize=14)
    ax.text(fig_w / 2, fig_h - TOP_MARGIN * 0.72,
            "top: token decoded for display (gray italic = no printable text, shown as its byte-level BPE symbol)"
            "   bottom: position:id", ha="center", va="center", fontsize=9, color="dimgray")

    y = fig_h - TOP_MARGIN
    for seq_idx, lines in enumerate(layout_by_seq):
        seq_top, seq_bottom = y, y - len(lines) * BOX_H - (len(lines) - 1) * ROW_GAP
        ax.text(LEFT_MARGIN - 0.15, (seq_top + seq_bottom) / 2, f"seq {seq_idx}",
                ha="right", va="center", fontsize=9, color="dimgray")
        for line in lines:
            for piece, id_str, is_bos, is_symbol, x, w in line:
                box_x = LEFT_MARGIN + x
                ax.add_patch(Rectangle((box_x, y - BOX_H), w, BOX_H, fill=is_bos, facecolor=BOS_FILL,
                                       edgecolor=BOS_EDGE if is_bos else TOKEN_EDGE, linewidth=0.9))
                ax.text(box_x + w / 2, y - 0.32 * BOX_H, piece, ha="center", va="center", fontsize=PIECE_FONT,
                        style="italic" if is_symbol else "normal", color="dimgray" if is_symbol else "black")
                ax.text(box_x + w / 2, y - 0.75 * BOX_H, id_str, ha="center", va="center", fontsize=ID_FONT, color="gray")
            y -= BOX_H + ROW_GAP
        y = seq_bottom - SEQ_GAP + ROW_GAP  # undo the extra ROW_GAP subtracted after the last line

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
