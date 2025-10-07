# PDAI Liquidity & Volatility Notes

_Date generated: 2025-10-07_

## 1. Routing Comparison (PDAI→DAI)

| Pair | Address | Reserve0 | Reserve1 | token0 | token1 |
|------|---------|---------:|---------:|--------|--------|
| PDAI/DAI | `0xfC64556FAA683e6087F425819C7Ca3C558e13aC1` | 14,580,480,752,943,932,644,221,843 | 49,024,792,185,948,612,161,113 | PDAI | DAI |
| PDAI/WPLS | `0xaE8429918FdBF9a5867e3243697637Dc56aa76A1` | 162,693,257,574,551,498,049,334,148 | 16,596,117,351,882,672,964,748,472,440 | PDAI | WPLS |
| WPLS/DAI | `0x146E1f1e060e5b5016Db0D118D2C5a11A240ae32` | 18,146,010,482,090,810,158,978,081,872 | 600,779,906,040,917,086,967,097 | WPLS | DAI |

_Reserves fetched live from PulseX via `collectors.reserve_fetcher`._

### Price Impact Example (swap 1,000 PDAI → DAI)

| Route | Output DAI | Effective Price (DAI/PDAI) | Effective Slippage* |
|-------|------------|----------------------------|---------------------|
| Direct PDAI→DAI | 3.3526 | 0.0033526 | 0.29 % |
| Via WPLS (PDAI→WPLS→DAI) | 3.3578 | 0.0033578 | 0.58 % |

#### Scaling the Trade Size

| Trade Size (PDAI) | Direct Output (DAI) | Direct Price | Direct Slippage | Routed Output (DAI) | Routed Price | Routed Slippage |
|-------------------|--------------------|-------------|-----------------|---------------------|-------------|------------------|
| 5 000 | 16.7573 | 0.0033515 | 0.32 % | 16.7766 | 0.0033553 | 0.58 % |
| 10 000 | 33.5032 | 0.0033503 | 0.36 % | 33.5513 | 0.0033551 | 0.59 % |
| 20 000 | 66.9606 | 0.0033480 | 0.43 % | 67.0947 | 0.0033547 | 0.60 % |
| 30 000 | 100.3723 | 0.0033457 | 0.49 % | 100.6303 | 0.0033543 | 0.61 % |
| 40 000 | 133.7384 | 0.0033435 | 0.56 % | 134.1580 | 0.0033539 | 0.63 % |
| 50 000 | 167.0591 | 0.0033412 | 0.63 % | 167.6778 | 0.0033536 | 0.64 % |

Note: amounts are quoted in token units (1 PDAI = 1e18 wei). Direct pool impact grows slowly with size because the pool is still large relative to these trades; routed path always pays two PulseX fees, so its effective slippage remains higher across the board.

\* Slippage measured against the instantaneous pool spot price. Both figures include PulseX’s 0.29 % fee per hop. The routed trade pays two fees, so even with much larger WPLS liquidity the net effective slippage is roughly double the direct pool.

**Takeaway:** With the current pool balances, the direct PDAI/DAI pool is liquid enough that routing through WPLS does _not_ improve execution—fees dominate any marginal gain. Stick with the direct pair unless the reserves change materially.

## 2. 2025‑06‑22 → 2025‑07‑04 Price Action

Source: `data/pdai_ohlcv_dai_730day_5m.csv`

- Window size: 3,457 five-minute bars
- Price (start): 0.0017976 DAI (2025‑06‑22 00:00 UTC)
- Price (peak): 0.0053153 DAI (2025‑07‑03 23:55 UTC)
- Price (end): 0.0053033 DAI (2025‑07‑04 00:00 UTC)
- Max/Min ratio: 3.11×
- Peak gain vs start: +195.68 %

So the dataset _does_ capture the ~200 % rally you mentioned. Any strategy tuned for PDAI should be able to exploit that kind of move—if it doesn’t, the issue is in the strategy logic/parameters rather than missing data.

## 3. Next Steps

1. Base your trade routing on the direct PDAI/DAI pool while it stays deeper than the routed alternative. Monitor reserves periodically.
2. Retune strategy parameter bounds with the confirmed volatility in mind (e.g., allow WaveTrend to trigger within ±8 – ±12 instead of ±60/−60).
3. Revisit walk-forward windows so the optimizer can “see” multi-week rallies like the June–July run.

_Let me know if you want this analysis expanded into an automated report script for future runs._
