# exp1ext predictions (recorded 2026-09-28, before any exp1ext data was generated)

Metric: sink_mass at the final query unless stated otherwise. "Common" = ids 256–999, "rare" = ids 1000–39999.

1. **Group 1** (sink_mass rises from rare to common, mostly late layers, e.g. L7H7, L10H11, L9H11, L11H1, L11H3):
   driven mainly by the **query** (final) token. In the 2×2 the query effect exceeds the context effect, and in the
   query sweep sink_mass is higher for common, low-id tokens than for rare ones.
2. **Group 2** (sink_mass falls from rare to common, early/middle layers, e.g. L2H4, L3H3, L6H8, L7H0, L2H0):
   driven mainly by the **context**, especially the previous token for heads that gain prev_mass (L2H4, L3H3, L6H8).
   Substituting a common token at position 31 into a rare sequence raises their prev_mass. L7H0 changes only
   through a drop in sink_mass, with prev_mass roughly unchanged.
3. **Byte-driven heads** (L0H6, L10H0, L9H8, which changed A→B only): per-token effects concentrate on ids 0–255.
4. **Controls** stay flat in every cell and for every token: L4H11 prev_mass ≈ 0.99, L5H1 Sink, L0H1 Self.
5. **Replication:** at least 12 of the 15 heads that changed in both exp1 comparisons again show label-mix
   TV ≥ 0.2 between the AA and CC cells on fresh sequences, with the same sink_mass direction; AA/CC per-head mean
   sink_mass correlates with exp1's A/C at r > 0.99 across all 144 heads.
