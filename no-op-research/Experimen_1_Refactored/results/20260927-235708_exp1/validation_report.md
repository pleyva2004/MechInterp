# Experiment 1 validation report

- Run: `20260927-235708_exp1`  |  model: `gpt2-small`  |  backends: TL (`gpt2-small`), HF (`gpt2`)  |  primary: TL
- 1000 sequences x 32 random tokens + BOS (50256) at position 0; key_len = 33, query index = 32; sampled without replacement within each sequence.
- Labels: Diffuse if max_weight < 0.2, else Sink (argmax 0) / Previous (argmax 31) / Self (argmax 32) / Other.

## Condition `A_rare` (ids 1000-39999)

### Checks

| check | backend | result | detail |
|---|---|---|---|
| 1. Attention rows sum to 1 | TL | PASS | max abs(sum - 1) = 4.24e-07 (atol 0.0001) |
| 1. Attention rows sum to 1 | HF | PASS | max abs(sum - 1) = 3.83e-07 (atol 0.0001) |
| 2. Determinism (first 50 re-extracted) | TL | PASS | bit-identical |
| 2. Determinism (first 50 re-extracted) | HF | PASS | bit-identical |
| 3. Known prev-token head L4H11 | TL | PASS | mode=Previous, prev_mass=0.988, consistency=0.997 |
| 3. Known prev-token head L4H11 | HF | PASS | mode=Previous, prev_mass=0.988, consistency=0.997 |
| 3. Known prev-token head L2H2 | TL | FAIL | mode=Previous, prev_mass=0.478, consistency=0.749 |
| 3. Known prev-token head L2H2 | HF | FAIL | mode=Previous, prev_mass=0.478, consistency=0.749 |
| 4. Head types by mode_label | TL | INFO | Diffuse 13.2%, Sink 73.6%, Previous 7.6%, Self 4.9%, Other 0.7% |
| 5. Sink mass higher in later layers | TL | PASS | layers 6-11: 0.605 vs layers 0-5: 0.331; peak L7 (0.704) |
| 6. Cross-library agreement | TL vs HF | PASS | max abs attn diff 3.69e-05 (atol 0.0001); per-sequence labels agree 100.000%; mode_label agrees 144/144 heads |

### 3. Known previous-token heads

Expected: mode_label = Previous with mean prev_mass >= 0.5.

| head | backend | mode_label | consistency | frac_previous | prev_mass mean | prev_mass std | sink_mass mean | result |
|---|---|---|---|---|---|---|---|---|
| L4H11 | TL | Previous | 0.997 | 0.997 | 0.988 | 0.059 | 0.000 | PASS |
| L4H11 | HF | Previous | 0.997 | 0.997 | 0.988 | 0.059 | 0.000 | PASS |
| L2H2 | TL | Previous | 0.749 | 0.749 | 0.478 | 0.189 | 0.079 | FAIL |
| L2H2 | HF | Previous | 0.749 | 0.749 | 0.478 | 0.189 | 0.079 | FAIL |

### 4. Head types (% of heads by mode_label)

Overall:

| mode_label | TL | HF |
|---|---|---|
| Diffuse | 13.2 | 13.2 |
| Sink | 73.6 | 73.6 |
| Previous | 7.6 | 7.6 |
| Self | 4.9 | 4.9 |
| Other | 0.7 | 0.7 |

Per layer (TL, % of the 12 heads in each layer):

| layer | Diffuse | Sink | Previous | Self | Other |
|---|---|---|---|---|---|
| 0 | 50.0 | 0.0 | 0.0 | 41.7 | 8.3 |
| 1 | 50.0 | 33.3 | 8.3 | 8.3 | 0.0 |
| 2 | 25.0 | 33.3 | 41.7 | 0.0 | 0.0 |
| 3 | 0.0 | 66.7 | 33.3 | 0.0 | 0.0 |
| 4 | 0.0 | 83.3 | 8.3 | 8.3 | 0.0 |
| 5 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 6 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 7 | 8.3 | 91.7 | 0.0 | 0.0 | 0.0 |
| 8 | 8.3 | 91.7 | 0.0 | 0.0 | 0.0 |
| 9 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 10 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| 11 | 16.7 | 83.3 | 0.0 | 0.0 | 0.0 |

Heads whose mode_label is not Sink (TL; consistency in parentheses):

- **Diffuse** (19): L0H0 (0.90), L0H2 (0.52), L0H6 (0.99), L0H8 (0.93), L0H9 (1.00), L0H11 (0.99), L1H1 (0.51), L1H2 (1.00), L1H4 (1.00), L1H5 (0.68), L1H7 (0.96), L1H10 (1.00), L2H1 (0.56), L2H7 (1.00), L2H10 (0.95), L7H7 (0.48), L8H6 (0.54), L11H0 (0.76), L11H8 (0.91)
- **Previous** (11): L1H0 (0.32), L2H2 (0.75), L2H3 (0.56), L2H5 (0.68), L2H8 (0.40), L2H9 (0.77), L3H2 (0.64), L3H6 (0.53), L3H7 (0.88), L3H8 (0.56), L4H11 (1.00)
- **Self** (7): L0H1 (1.00), L0H3 (1.00), L0H4 (1.00), L0H5 (1.00), L0H10 (0.82), L1H11 (1.00), L4H7 (0.82)
- **Other** (1): L0H7 (0.39)

### 5. Mean sink_mass per layer

Mean over the layer's heads of each head's mean sink_mass (attention on BOS).

| layer | TL | HF |
|---|---|---|
| 0 | 0.068 | 0.068 |
| 1 | 0.156 | 0.156 |
| 2 | 0.209 | 0.209 |
| 3 | 0.368 | 0.368 |
| 4 | 0.528 | 0.528 |
| 5 | 0.657 | 0.657 |
| 6 | 0.609 | 0.609 |
| 7 | 0.704 | 0.704 |
| 8 | 0.653 | 0.653 |
| 9 | 0.681 | 0.681 |
| 10 | 0.555 | 0.555 |
| 11 | 0.426 | 0.426 |

### 6. Cross-library agreement (TransformerLens vs Hugging Face)

- Same token-ID tensors fed to both; max abs attention difference: 3.69e-05
- Per-(sequence, layer, head) labels agreeing: 100.000%
- Per-head mode_label agreeing: 144/144

### Figures

- Example input sequences: [A_rare_sequences](figures/A_rare_sequences.png)
- TL: [A_rare_mode_label](figures/A_rare_mode_label.png), [A_rare_sink_mass](figures/A_rare_sink_mass.png), [A_rare_entropy](figures/A_rare_entropy.png)
- HF: [A_rare_hf_mode_label](figures/A_rare_hf_mode_label.png), [A_rare_hf_sink_mass](figures/A_rare_hf_sink_mass.png), [A_rare_hf_entropy](figures/A_rare_hf_entropy.png)
