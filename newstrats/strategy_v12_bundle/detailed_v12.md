# v12 (Unsliced, Aggressive) — Detailed Summary

Aggressive neighbor sweep around the v11 winner. All costs unsliced (single-shot per side, step-rounded).

## Results table (sorted by return)
| name            |   e_days |   x_days |   ema_days |   dd_base |    k |    w |   dd_min |   dd_max | entry_ema_gate   |   final_equity |   return_pct |   trades |
|:----------------|---------:|---------:|-----------:|----------:|-----:|-----:|---------:|---------:|:-----------------|---------------:|-------------:|---------:|
| v11_best        |       12 |        2 |          3 |      0.13 | 0.62 | 0.1  |     0.1  |     0.45 | False            |       21796.3  |     2079.63  |       32 |
| A5_highK        |       12 |        2 |          3 |      0.13 | 0.7  | 0.12 |     0.1  |     0.5  | False            |       20046.1  |     1904.61  |       32 |
| A8_e11_ema4     |       11 |        2 |          4 |      0.12 | 0.65 | 0.12 |     0.1  |     0.55 | False            |       17347.4  |     1634.74  |       35 |
| A3_e11          |       11 |        2 |          3 |      0.12 | 0.65 | 0.15 |     0.1  |     0.55 | False            |       16440.3  |     1544.03  |       35 |
| A6_highW        |       12 |        2 |          3 |      0.12 | 0.62 | 0.15 |     0.1  |     0.5  | False            |       15187.4  |     1418.74  |       34 |
| A1_looserDD     |       12 |        2 |          3 |      0.12 | 0.65 | 0.12 |     0.1  |     0.5  | False            |        9597.55 |      859.755 |       37 |
| A4_fastEMAexit  |       12 |        2 |          2 |      0.12 | 0.65 | 0.1  |     0.1  |     0.5  | False            |        8957.27 |      795.727 |       39 |
| A2_loosestDD    |       12 |        2 |          3 |      0.1  | 0.7  | 0.12 |     0.08 |     0.55 | False            |        7034.64 |      603.464 |       45 |
| A7_x1_quickexit |       12 |        1 |          3 |      0.12 | 0.62 | 0.12 |     0.1  |     0.5  | False            |        6326.92 |      532.692 |       42 |

## Chosen params
{'e_days': 12, 'x_days': 2, 'ema_days': 3, 'dd_base': 0.13, 'k': 0.62, 'w': 0.1, 'dd_min': 0.1, 'dd_max': 0.45, 'entry_ema_gate': False}

**Return:** 2079.63% | **Final:** 21796.33 DAI | **Trades:** 32
