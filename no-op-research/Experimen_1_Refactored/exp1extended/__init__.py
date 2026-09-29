"""Experiment 1-extended: attribute the rare-vs-common head shifts of Experiment 1 to positions and tokens.

Stage 0 routes each target head (query-side, key-side, context) from the saved Exp 1 sequences; Stage 1 is a
paired position factorial with a surface-matched rare pool; Stage 2 sweeps single tokens through the final
position. Reuses exp1's sampling, backends, metrics and statistics. Config: configs/exp1extended.yaml.
"""
