**Short answer:** the biggest shifts are at **L7H7, L10H11, L2H0 and L2H4**, each with TV ≈ 0.43–0.51. Both graphs pick out largely the same heads: across all 144 heads, the A → B and A → C shifts correlate at r = 0.85, and 15 heads clear 0.2 in both.

Top 10, ranked by average shift across the two comparisons. "Label mix" gives each head's top labels as a percentage of sequences; A and B/C columns are separated by an arrow.

| head | TV A→B | TV A→C | label mix A → B / C | sink_mass A → B / C | what changed |
|---|---|---|---|---|---|
| **L7H7** | 0.51 | 0.51 | Diff 48, Sink 42 → Sink 93 / 93 | 0.23 → 0.54 / 0.54 | spreads out on rare tokens, rests on BOS on common ones |
| **L10H11** | 0.44 | 0.52 | Sink 41, Diff 37 → Sink 85 / 93 | 0.25 → 0.47 / 0.55 | same |
| **L2H0** | 0.47 | 0.48 | Sink 58 → Other 30–38, Prev 26, Diff 17–24 | 0.25 → 0.14 / 0.14 | stops resting on common tokens and scatters across labels |
| **L2H4** | 0.42 | 0.45 | Sink 96 → Sink ~52, Prev 23, Other 22–25 | 0.71 → 0.36 / 0.36 | stops resting on common tokens; prev_mass 0.09 → 0.20 |
| L1H7 | 0.24 | 0.47 | Diff 96 → Diff 72 / 49, Sink 25 / 51 | 0.11 → 0.17 / 0.20 | becomes partly Sink |
| L6H8 | 0.40 | 0.29 | Sink 70 → Prev 66 / 52 | 0.53 → 0.30 / 0.34 | Sink → Prev; prev_mass 0.33 → 0.43 / 0.38 |
| L7H0 | 0.32 | 0.35 | Sink 80 → Sink ~46, Prev ~37 | 0.57 → 0.33 / 0.31 | Sink → Prev, but prev_mass barely moves (see below) |
| L3H3 | 0.37 | 0.28 | Sink 76 → Prev 50 / 45 | 0.59 → 0.33 / 0.38 | Sink → Prev; prev_mass 0.24 → 0.39 |
| L1H5 | 0.27 | 0.37 | Diff 68 → Sink 47 / 59 | 0.16 → 0.20 / 0.23 | becomes partly Sink |
| L9H11 | 0.32 | 0.30 | Sink 49, Diff 35 → Sink 80 / 79 | 0.28 → 0.43 / 0.40 | spreads out on rare tokens |

## Two groups moving in opposite directions

- **More active on rare tokens (Diff → Sink): mostly late layers.** L7H7, L10H11, L9H11, L11H3 and L11H1 spread their attention on rare-token sequences: entropy is higher, e.g. L7H7 2.70 vs 1.96. On common tokens they fall back to BOS. L1H5 and L1H7 do the same in layer 1.
- **More active on common tokens (Sink → Prev/Other): early and middle layers.** L2H4, L3H3, L6H8, L7H0 and L2H0, plus smaller ones (L10H9, L5H2, L5H6, L4H0), rest on BOS on rare tokens and pick up previous-token or scattered attention on common ones.

## Caveats

- **"Sink → Prev" is not always new previous-token attention.** For L2H4, L3H3, L6H8 and L10H9, prev_mass genuinely rises. For L7H0, prev_mass only goes from 0.26 to 0.29: sink mass fell below it, so Prev wins the label more often.
- **Some heads show up in only one graph:**
  - A → B only: **L0H6** (0.34 vs 0.05), **L10H0** (0.23 vs 0.02) and L9H8 (0.23 vs 0.09) move only when byte tokens are present, so B's contents drive them rather than rare vs common.
  - A → C only: L8H2 (0.05 vs 0.25) and L11H9 (0.18 vs 0.36) move more against the byte-free set.

To see what the rare-vs-common effect itself does, rely on the heads that appear in both graphs.

I computed these numbers in a scratch script from the saved attention arrays, not from a pipeline output. I can add the table to the Phase 2 report if you want it kept with the run.
