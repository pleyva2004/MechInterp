# Pre-registered predictions: TDA analysis 3, 1-D persistence of each head's distance curve

Written 2026-10-25, before any persistence was computed on these curves. What I had looked at when writing: the shapes
and keys of `controlled.npz` and `capital.npz`, and `exp4/LOG.md` (the verified findings quoted below). I had not
plotted or printed any D(d) curve for this analysis. Code: `exp4/tda3_curves.py`. No model run; saved arrays only.

The purpose is to replace eyeballed curve shapes ("decays", "flat", "spike at the far end", "rises with distance")
with a count and ranking of each curve's peaks and troughs that survives sampling noise.

## Curves

Curve f(d), d = 1..31 (d = 32 − τ, array index d − 1), is the attention-to-τ difference D(d) of the **country arm**, for
every head (144) × condition × filler type {common, rare}:

| condition | what | observed curve (saved) | per-row curves (for the null) |
|---|---|---|---|
| random | random final token (Exp 4) | `{b}/country/mean` in `20261024-000548_exp4ctrl/controlled.npz` | `row_differences` of `{b}/country_masses` and `{b}/control` |
| own | final token = the country's capital (Exp 4b) | `own/{b}/country/mean` in `20261024-150407_exp4cap/capital.npz` | same, from the float16 per-row arrays |
| wrong | final token = a derangement's capital | `wrong/{b}/country/mean` | same |
| fact | own − wrong (extra, not in the named checks) | `{b}/fact/mean` | per-row own − per-row wrong |

That is 144 × 4 × 2 = 1152 curves. The observed curve is always the saved mean (exact); the per-row arrays are used
only to build the null, and I report max |recomputed row-mean − saved mean| as a check (float16 storage for the
capital runs).

## Persistence conventions

- **Filtration.** Values sit on the 31 points of a path graph; an edge enters at the larger of its two end values.
  This is the V-construction, so the two endpoints are ordinary vertices with a single neighbour: an endpoint higher than
  its neighbour is a peak, lower is a trough. (gudhi's `top_dimensional_cells` constructor would put the values on
  edges and distort the endpoints, so it is not used.)
- **Troughs** = sublevel-set H0 bars of f. **Peaks** = sublevel-set H0 bars of −f (superlevel H0 of f). A bar's
  location is the d of the vertex that gave birth to it; persistence = |death − birth|.
- **Essential bars.** The global minimum (trough) and the global maximum (peak) never die; their persistence is
  defined as the curve's range R = max f − min f (the usual pairing in extended persistence). So every curve with
  R > 0 has one essential peak and one essential trough, both with persistence R. All other bars are non-essential.
- The implementation is my own vectorised elder-rule merge on the path graph (fast enough for ~10^6 null curves). It is
  validated against gudhi's `SimplexTree` (vertices at f, edges at the max of their ends) on random curves with exact
  bar equality, and on hand-built curves whose endpoints are extrema.

## Noise threshold (per head × condition × filler)

Null: resample the 200 rows with replacement (B = 2000, seed 0, the same resample counts reused across heads of one
condition × filler), **subtract the full-sample mean curve**, recompute the mean curve. Each null curve is the sampling
noise of the mean curve with the true shape removed and the per-row covariance across d kept. Per null curve compute

- the range R*, and
- M* = the largest **non-essential** bar over peaks and troughs.

Thresholds, primary at the 99th percentile (sensitivity at the 95th):

- θ_ess = q99 of R* (applies to the essential bars),
- θ_non = q99 of M* (applies to non-essential bars).

**Significance of a bar.** Persistence > its θ **and** persistence ≥ 0.02 (the repo's `margin`, so a bar can be
statistically clear but too small to matter; the margin-free count is reported as a sensitivity). **Gate:** no bar of a
curve is significant unless the curve's range itself passes (R > θ_ess and R ≥ 0.02).

## Features and sign

Sign s = + if (max f − median f) ≥ (median f − min f), else −; the dominant extremum is the global max when s = + and the
global min when s = −. The opposite global extremum is the curve's baseline, not a feature (otherwise every
dip-then-flat curve would also have a "significant peak" at an arbitrary flat point, with persistence equal to the dip).
The **feature set F** of a curve = {the dominant essential extremum} ∪ {every significant non-essential peak and
trough}, all subject to the gate.

## Shape taxonomy (operational, first matching rule wins)

Let ρ = R / max|f|, d* = location of the dominant extremum, and P = the significant peaks of s·f (for a negative curve
a "peak" is a dip). Zones: near d ≤ 3, far d ≥ 25, mid 4–24.

1. **null:** gate fails and |mean_d f| < 0.02.
2. **plateau:** (gate fails, or ρ < 0.5) and |mean_d f| ≥ 0.02. ρ < 0.5 means the curve varies by less than half its
   own height, so a slow slide such as 0.43 → 0.35 is a plateau, not a decay.
3. **U-shape:** P has a peak in the near zone and one in the far zone.
4. **multi-modal:** |P| ≥ 2 and not U-shape.
5. **near-peak decay:** |P| = 1 and d* ≤ 3.
6. **far rise:** |P| = 1 and d* ≥ 25.
7. **mid-peak:** |P| = 1 and 4 ≤ d* ≤ 24.

The class is sign-agnostic and the sign is recorded separately ("near-peak decay" with s = − is a near dip that fades).

## Named checks (pass/fail; each must hold in both filler types)

| id | claim | operational test |
|---|---|---|
| C1 | L9H8's rise at d = 31 (country right after BOS) is a significant peak, random final token | In the random condition, the superlevel bar born at d = 31 is significant (essential if d = 31 is the global max, else non-essential), for common and rare |
| C2 | L11H8 is a single near-peak decay under all three conditions | Class = near-peak decay with s = + for random, own and wrong, both fillers (6 curves) |
| C3 | L3H0 with its own capital has its dominant peak at the far end | Own condition: s = +, gate passes, d* ≥ 25, for common and rare |
| C4 | L0H2 with its own capital is a plateau | Own condition: class = plateau, for common and rare. Reported alongside: the number of significant bars in F at 2 ≤ d ≤ 30 |
| C5 | Exp 1 previous-token heads (L1H0, L2H2, L2H3, L2H5, L2H8, L2H9, L3H2, L3H6, L3H7, L3H8, L4H11) have no significant feature beyond d = 8 under any condition | No bar in F located at d ≥ 9, for each of these 11 heads, random / own / wrong, common and rare (66 curves). Any violation is listed |

My priors before seeing the numbers (so the verdicts are not fitted afterwards):

- **C1 PASS, 60%.** The d = 31 rise is the visible far-end feature in the Exp 4 viewer, but it may be a small bump, and the
  99th-percentile threshold could remove it in the rare filler.
- **C2 PASS, 45%.** The random-token curve should be one near peak. With a capital query L11H8 was meaningful at 25/30
  distances, so the tail is high and a secondary significant bump (multi-modal) is the likely failure.
- **C3 PASS, 60%.** It is an association head with a positive fact at all 30 distances; whether the maximum is at the
  far end rather than at d = 1 is unknown.
- **C4 PASS, 55%.** L0H2 attends 0.43 / 0.40 at d = 1 and 0.38 / 0.35 at d = 20, so ρ is about 0.2 if nothing else happens;
  the risk is a significant interior wiggle of at least 0.02, which would not change ρ but would flip the strict reading.
- **C5 FAIL, 65%** (PASS 35%). Exp 4b's F6 already failed narrowly (L1H0 at d = 9, common), and the own / wrong conditions
  add more chances for one bar at d ≥ 9 among 66 curves.

## Descriptive expectations (not pass/fail; scored in the report)

- **E1.** Most of the 144 heads are null or plateau in the random condition (more than half of them, in both fillers).
- **E2.** The own condition has more heads with at least one significant feature than the random condition (the
  association heads L0H2, L1H11, L3H0, L4H10 appear), in both fillers.
- **E3.** The four late any-distance heads of Exp 4 (L9H8, L10H3, L11H7, L11H2) do not all share one shape class.

## Sensitivity runs (descriptive)

Class counts are recomputed with (a) the 95th-percentile thresholds, (b) no 0.02 margin, (c) ρ cut-off 0.3 and 0.7. A
verdict that flips under any of these is flagged as fragile in the report.

## Checks (not predictions)

- **Algorithm:** bars equal gudhi's `SimplexTree` persistence on 2000 random curves of length 31, and on hand-built
  curves with extrema at the endpoints (ties included).
- **Data:** per condition × filler, max |row-mean of recomputed D − saved mean| over all 31 × 144 cells (expect ≤ 1e-6 for
  float32 runs and ≤ 5e-4 for the float16 capital arrays).
- **Null calibration:** on a fresh bootstrap of 500 curves per head (a different seed), the share whose range exceeds θ_ess
  is about 1% (q99) and 5% (q95).
- **Expected false positives:** at the 99th percentile about 1% of the 1152 curves, roughly 11, pass the gate by chance and
  carry a spurious feature; the report states the observed count of low-level curves against that.

## Known weaknesses, stated in advance

- Each curve has 31 points, so peaks closer than a few points are barely resolvable, and "interior" has only 29 positions.
- The null assumes the rows are exchangeable and that the noise of the mean curve is the same whether or not a real
  shape is present; the per-row variance is largest where the effect is, so the null is conservative for strong heads.
- The 11 countries are averaged inside each row, so the row-level noise includes variation from the 11 fixed pairs that is
  not sampling noise; Exp 4's CIs have the same limitation.
- 0.02, 0.5 and the zone borders (3, 25) are chosen a priori, not tuned; the sensitivity runs show how much they matter.
