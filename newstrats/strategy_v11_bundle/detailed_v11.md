# v11 (Unsliced) — Detailed Summary

We tested two targeted neighbors around the v10 winner.

Results (sorted by return):
| name              |   e_days |   x_days |   ema_days |   dd_base |    k |   w |   dd_min |   dd_max | entry_ema_gate   |   final_equity |   return_pct |   trades |
|:------------------|---------:|---------:|-----------:|----------:|-----:|----:|---------:|---------:|:-----------------|---------------:|-------------:|---------:|
| dd_base0.13_k0.62 |       12 |        2 |          3 |      0.13 | 0.62 | 0.1 |      0.1 |     0.45 | False            |        21796.3 |      2079.63 |       32 |
| v10_best          |       12 |        2 |          3 |      0.14 | 0.6  | 0.1 |      0.1 |     0.45 | False            |        20752.9 |      1975.29 |       31 |
| k0.62             |       12 |        2 |          3 |      0.14 | 0.62 | 0.1 |      0.1 |     0.45 | False            |        20752.9 |      1975.29 |       31 |

**Chosen params:** {'e_days': 12, 'x_days': 2, 'ema_days': 3, 'dd_base': 0.13, 'k': 0.62, 'w': 0.1, 'dd_min': 0.1, 'dd_max': 0.45, 'entry_ema_gate': False}
**Return:** 2079.63% | **Final:** 21796.33 DAI | **Trades:** 32
