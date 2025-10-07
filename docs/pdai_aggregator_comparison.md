# PDAI Aggregator Comparison (PulseX vs. Piteas)

_Date: 2025-10-07_

## Scenario

- Trade size: **$50,000 worth of PDAI**, quoted in DAI on PulseChain.
- Source data: live quotes captured from PulseX v1.1.3 and Piteas (v2.0.6) UIs at the same block range.
- Goal: compare the routing strategy, expected output, and effective slippage for both aggregators on a large buy and the corresponding sell of the acquired PDAI.

## Snapshot: 50,000 DAI ➜ PDAI

| Aggregator | Expected Output (PDAI) | Price Impact | Notes on Route |
|------------|------------------------|--------------|----------------|
| **PulseX** | **13,518,103** | **−8.44 %** | Router splits the order into three pipelines: 80 % via `DAI → WPLS (v1) → PDAI (v2)`, 10 % via `DAI → Stable3 → WPLS (v1) → PDAI (v1)`, and 10 % via a direct `DAI → PDAI (v2)` path. All hops incur 0.29 % fee each. |
| **Piteas** | **13,730,192** | **−7.20 %** | Pathfinder allocates ~46 % through `DAI → WPLS` (mix of PulseX v1/v2/9mm v3) then to PDAI across PulseX v2 + pDex v3; ~35 % via `DAI → USDC` (PulseX Stable + Phux) then `USDC → WPLS → PDAI`; ~19 % via `DAI → WPLS` followed by `WPLS → pWBTC → PDAI`. Fees vary per venue (0.25–0.30 %). |

**Observation:** Piteas stretches the order across additional venues (Phux, pDex, 9mm) and stable routing, reducing slippage by ~1.2 % and delivering ~212 k more PDAI for the same 50 k DAI compared with PulseX smart routing.

## Snapshot: Sell the Acquired PDAI

Using the 13.5–13.7 M PDAI acquired above as the reference size (PulseX quote shows up to 15 M PDAI), we captured the closest available sell quotes.

| Aggregator | Input (PDAI) | Expected DAI Out | Price Impact | Notes |
|------------|--------------|------------------|--------------|-------|
| **PulseX** | 15,000,000 | 45,957.4 DAI | −8.11 % | Route splits similarly: majority through WPLS pools plus a direct PDAI/DAI v2 hop. |
| **Piteas** | 15,000,000 | 46,331.7 DAI | −7.20 % | Reverse path mirrors the buy distribution, leveraging PulseX Stable, Phux, pDex, and 9mm pools. |

**Observation:** The relative advantage persists on the exit: Piteas returns ~374 DAI more on a 15 M PDAI liquidation (~0.83 % improvement) by leaning on alternative pools.

## Takeaways

1. **Routing diversity matters.** PulseX’s built-in smart router already splits the order across three pipelines, but it only taps PulseX venues. Piteas widens the search space to include Phux, pDex, 9mm, etc., which adds extra depth.
2. **Slippage gap (~1 % absolute) is consistent for both entry and exit** at the 50 k USD size. If the bot can interface with Piteas (or emulate its route search), it should improve fill quality.
3. **Performance vs. complexity.** Piteas’ pathfinder constantly samples multiple pools every block. Replicating this in-house would require a routing engine; otherwise, consider calling their API when trades exceed a size threshold.

## Next Steps (Suggested)

- Integrate a routing module that can query Piteas’ API v2.3 for large trades while falling back to PulseX router for small orders.
- Add slippage alerts that compare current on-chain reserves with the latest aggregator quotes so the optimizer knows when a strategy would have been unfillable due to price impact.
