# Experiment 1 validation report

- Run: `20260928-004336_exp1`  |  model: `gpt2-small`  |  backends: TL (`gpt2-small`), HF (`gpt2`)  |  primary: TL
- 1000 sequences x 32 random tokens + BOS (50256) at position 0; key_len = 33, query index = 32; sampled without replacement within each sequence.
- Labels: Diffuse if max_weight < 0.2, else Sink (argmax 0) / Previous (argmax 31) / Self (argmax 32) / Other.

## Condition `B_common` (ids 0-999)

### Checks

| check | backend | result | detail |
|---|---|---|---|
| 1. Attention rows sum to 1 | TL | PASS | max abs(sum - 1) = 3.79e-07 (atol 0.0001) |
| 1. Attention rows sum to 1 | HF | PASS | max abs(sum - 1) = 4.15e-07 (atol 0.0001) |
| 2. Determinism (first 50 re-extracted) | TL | PASS | bit-identical |
| 2. Determinism (first 50 re-extracted) | HF | PASS | bit-identical |
| 3. Known prev-token head L4H11 | TL | PASS | mode=Previous, prev_mass=0.984, consistency=0.988 |
| 3. Known prev-token head L4H11 | HF | PASS | mode=Previous, prev_mass=0.984, consistency=0.988 |
| 3. Known prev-token head L2H2 | TL | PASS | mode=Previous, prev_mass=0.504, consistency=0.733 |
| 3. Known prev-token head L2H2 | HF | PASS | mode=Previous, prev_mass=0.504, consistency=0.733 |
| 4. Head types by mode_label | TL | INFO | Diffuse 10.4%, Sink 73.6%, Previous 9.0%, Self 4.9%, Other 2.1% |
| 5. Sink mass higher in later layers | TL | PASS | layers 6-11: 0.572 vs layers 0-5: 0.313; peak L9 (0.671) |
| 6. Cross-library agreement | TL vs HF | PASS | max abs attn diff 2.11e-05 (atol 0.0001); per-sequence labels agree 100.000%; mode_label agrees 144/144 heads |

### 3. Known previous-token heads

Expected: mode_label = Previous with mean prev_mass >= 0.5.

| head | backend | mode_label | consistency | frac_previous | prev_mass mean | prev_mass std | sink_mass mean | result |
|---|---|---|---|---|---|---|---|---|
| L4H11 | TL | Previous | 0.988 | 0.988 | 0.984 | 0.102 | 0.000 | PASS |
| L4H11 | HF | Previous | 0.988 | 0.988 | 0.984 | 0.102 | 0.000 | PASS |
| L2H2 | TL | Previous | 0.733 | 0.733 | 0.504 | 0.232 | 0.028 | PASS |
| L2H2 | HF | Previous | 0.733 | 0.733 | 0.504 | 0.232 | 0.028 | PASS |

### 4. Head types (% of heads by mode_label)

Overall:

| mode_label | TL | HF |
|---|---|---|
| Diffuse | 10.4 | 10.4 |
| Sink | 73.6 | 73.6 |
| Previous | 9.0 | 9.0 |
| Self | 4.9 | 4.9 |
| Other | 2.1 | 2.1 |

Per layer (TL, % of the 12 heads in each layer):

| layer | Diffuse | Sink | Previous | Self | Other |
|---|---|---|---|---|---|
| 0 | 41.7 | 8.3 | 0.0 | 41.7 | 8.3 |
| 1 | 50.0 | 33.3 | 0.0 | 8.3 | 8.3 |
| 2 | 16.7 | 33.3 | 41.7 | 0.0 | 8.3 |
| 3 | 0.0 | 58.3 | 41.7 | 0.0 | 0.0 |
| 4 | 0.0 | 83.3 | 8.3 | 8.3 | 0.0 |
| 5 | 0.0 | 91.7 | 8.3 | 0.0 | 0.0 |
| 6 | 0.0 | 91.7 | 8.3 | 0.0 | 0.0 |
| 7 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 8 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 9 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 10 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 11 | 16.7 | 83.3 | 0.0 | 0.0 | 0.0 |

Heads whose mode_label is not Sink (TL; consistency in parentheses):

- **Diffuse** (15): L0H0 (0.90), L0H6 (0.65), L0H8 (0.83), L0H9 (0.98), L0H11 (0.86), L1H1 (0.37), L1H2 (1.00), L1H4 (0.99), L1H7 (0.72), L1H9 (0.57), L1H10 (1.00), L2H7 (1.00), L2H10 (0.98), L11H0 (0.76), L11H8 (0.73)
- **Previous** (13): L2H2 (0.73), L2H3 (0.41), L2H5 (0.55), L2H8 (0.46), L2H9 (0.67), L3H2 (0.57), L3H3 (0.50), L3H6 (0.45), L3H7 (0.78), L3H8 (0.42), L4H11 (0.99), L5H6 (0.49), L6H8 (0.66)
- **Self** (7): L0H1 (0.97), L0H3 (0.99), L0H4 (0.92), L0H5 (0.95), L0H10 (0.64), L1H11 (0.96), L4H7 (0.80)
- **Other** (3): L0H7 (0.42), L1H0 (0.37), L2H0 (0.30)

### 5. Mean sink_mass per layer

Mean over the layer's heads of each head's mean sink_mass (attention on BOS).

| layer | TL | HF |
|---|---|---|
| 0 | 0.075 | 0.075 |
| 1 | 0.159 | 0.159 |
| 2 | 0.168 | 0.168 |
| 3 | 0.364 | 0.364 |
| 4 | 0.484 | 0.484 |
| 5 | 0.627 | 0.627 |
| 6 | 0.552 | 0.552 |
| 7 | 0.664 | 0.664 |
| 8 | 0.639 | 0.639 |
| 9 | 0.671 | 0.671 |
| 10 | 0.502 | 0.502 |
| 11 | 0.404 | 0.404 |

### 6. Cross-library agreement (TransformerLens vs Hugging Face)

- Same token-ID tensors fed to both; max abs attention difference: 2.11e-05
- Per-(sequence, layer, head) labels agreeing: 100.000%
- Per-head mode_label agreeing: 144/144

### Figures

- Example input sequences: [B_common_sequences](figures/B_common_sequences.png)
- TL: [B_common_mode_label](figures/B_common_mode_label.png), [B_common_sink_mass](figures/B_common_sink_mass.png), [B_common_entropy](figures/B_common_entropy.png)
- HF: [B_common_hf_mode_label](figures/B_common_hf_mode_label.png), [B_common_hf_sink_mass](figures/B_common_hf_sink_mass.png), [B_common_hf_entropy](figures/B_common_hf_entropy.png)

## Condition `C_common_clean` (ids 256-999)

### Checks

| check | backend | result | detail |
|---|---|---|---|
| 1. Attention rows sum to 1 | TL | PASS | max abs(sum - 1) = 3.95e-07 (atol 0.0001) |
| 1. Attention rows sum to 1 | HF | PASS | max abs(sum - 1) = 4.16e-07 (atol 0.0001) |
| 2. Determinism (first 50 re-extracted) | TL | PASS | bit-identical |
| 2. Determinism (first 50 re-extracted) | HF | PASS | bit-identical |
| 3. Known prev-token head L4H11 | TL | PASS | mode=Previous, prev_mass=0.986, consistency=0.992 |
| 3. Known prev-token head L4H11 | HF | PASS | mode=Previous, prev_mass=0.986, consistency=0.992 |
| 3. Known prev-token head L2H2 | TL | FAIL | mode=Previous, prev_mass=0.496, consistency=0.696 |
| 3. Known prev-token head L2H2 | HF | FAIL | mode=Previous, prev_mass=0.496, consistency=0.696 |
| 4. Head types by mode_label | TL | INFO | Diffuse 10.4%, Sink 75.7%, Previous 6.2%, Self 4.9%, Other 2.8% |
| 5. Sink mass higher in later layers | TL | PASS | layers 6-11: 0.594 vs layers 0-5: 0.327; peak L9 (0.688) |
| 6. Cross-library agreement | TL vs HF | PASS | max abs attn diff 2.38e-05 (atol 0.0001); per-sequence labels agree 100.000%; mode_label agrees 144/144 heads |

### 3. Known previous-token heads

Expected: mode_label = Previous with mean prev_mass >= 0.5.

| head | backend | mode_label | consistency | frac_previous | prev_mass mean | prev_mass std | sink_mass mean | result |
|---|---|---|---|---|---|---|---|---|
| L4H11 | TL | Previous | 0.992 | 0.992 | 0.986 | 0.078 | 0.000 | PASS |
| L4H11 | HF | Previous | 0.992 | 0.992 | 0.986 | 0.078 | 0.000 | PASS |
| L2H2 | TL | Previous | 0.696 | 0.696 | 0.496 | 0.233 | 0.037 | FAIL |
| L2H2 | HF | Previous | 0.696 | 0.696 | 0.496 | 0.233 | 0.037 | FAIL |

### 4. Head types (% of heads by mode_label)

Overall:

| mode_label | TL | HF |
|---|---|---|
| Diffuse | 10.4 | 10.4 |
| Sink | 75.7 | 75.7 |
| Previous | 6.2 | 6.2 |
| Self | 4.9 | 4.9 |
| Other | 2.8 | 2.8 |

Per layer (TL, % of the 12 heads in each layer):

| layer | Diffuse | Sink | Previous | Self | Other |
|---|---|---|---|---|---|
| 0 | 41.7 | 8.3 | 0.0 | 41.7 | 8.3 |
| 1 | 41.7 | 41.7 | 0.0 | 8.3 | 8.3 |
| 2 | 16.7 | 33.3 | 33.3 | 0.0 | 16.7 |
| 3 | 0.0 | 75.0 | 25.0 | 0.0 | 0.0 |
| 4 | 0.0 | 83.3 | 8.3 | 8.3 | 0.0 |
| 5 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 6 | 0.0 | 91.7 | 8.3 | 0.0 | 0.0 |
| 7 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 8 | 8.3 | 91.7 | 0.0 | 0.0 | 0.0 |
| 9 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 10 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 11 | 16.7 | 83.3 | 0.0 | 0.0 | 0.0 |

Heads whose mode_label is not Sink (TL; consistency in parentheses):

- **Diffuse** (15): L0H0 (0.89), L0H6 (0.94), L0H8 (0.91), L0H9 (0.99), L0H11 (0.84), L1H1 (0.46), L1H2 (1.00), L1H4 (1.00), L1H9 (0.57), L1H10 (1.00), L2H7 (1.00), L2H10 (0.99), L8H6 (0.51), L11H0 (0.68), L11H8 (0.81)
- **Previous** (9): L2H2 (0.70), L2H5 (0.62), L2H8 (0.49), L2H9 (0.73), L3H2 (0.53), L3H6 (0.41), L3H7 (0.80), L4H11 (0.99), L6H8 (0.52)
- **Self** (7): L0H1 (1.00), L0H3 (0.98), L0H4 (0.92), L0H5 (1.00), L0H10 (0.68), L1H11 (1.00), L4H7 (0.77)
- **Other** (4): L0H7 (0.40), L1H0 (0.49), L2H0 (0.38), L2H3 (0.43)

### 5. Mean sink_mass per layer

Mean over the layer's heads of each head's mean sink_mass (attention on BOS).

| layer | TL | HF |
|---|---|---|
| 0 | 0.073 | 0.073 |
| 1 | 0.167 | 0.167 |
| 2 | 0.174 | 0.174 |
| 3 | 0.410 | 0.410 |
| 4 | 0.492 | 0.492 |
| 5 | 0.647 | 0.647 |
| 6 | 0.567 | 0.567 |
| 7 | 0.674 | 0.674 |
| 8 | 0.617 | 0.617 |
| 9 | 0.688 | 0.688 |
| 10 | 0.586 | 0.586 |
| 11 | 0.433 | 0.433 |

### 6. Cross-library agreement (TransformerLens vs Hugging Face)

- Same token-ID tensors fed to both; max abs attention difference: 2.38e-05
- Per-(sequence, layer, head) labels agreeing: 100.000%
- Per-head mode_label agreeing: 144/144

### Figures

- Example input sequences: [C_common_clean_sequences](figures/C_common_clean_sequences.png)
- TL: [C_common_clean_mode_label](figures/C_common_clean_mode_label.png), [C_common_clean_sink_mass](figures/C_common_clean_sink_mass.png), [C_common_clean_entropy](figures/C_common_clean_entropy.png)
- HF: [C_common_clean_hf_mode_label](figures/C_common_clean_hf_mode_label.png), [C_common_clean_hf_sink_mass](figures/C_common_clean_hf_sink_mass.png), [C_common_clean_hf_entropy](figures/C_common_clean_hf_entropy.png)
