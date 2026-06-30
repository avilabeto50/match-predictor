# Knockout Stage Model Comparison Report
*Generated: 2026-06-30 00:27:16*

## Overview

- **Frozen model**: fitted on 1,433 historical matches (2014–2025)
- **Re-fitted model**: fitted on 1,505 matches (historical + 72 group stage results)
- **Group stage weight**: 1.0 (World Cup tier)

## Log-Likelihood Comparison

| Metric | Value |
|--------|-------|
| Frozen model NLL (on historical data) | 2031.58 |
| Frozen model NLL (on combined data) | 2243.57 |
| Re-fitted model NLL (on combined data) | 2237.91 |
| Log-likelihood improvement from re-fit | **5.66** (↑ better) |

## Model Parameters

| Parameter | Frozen | Re-Fitted | Change |
|-----------|--------|-----------|--------|
| μ (global rate) | 1.2189 | 1.2285 | +0.0096 |
| α mean (attack) | 1.0406 | 1.0457 | +0.0051 |
| β mean (defense) | 1.0306 | 1.0359 | +0.0053 |
| α std | 0.2966 | 0.3136 | — |
| β std | 0.2568 | 0.2807 | — |

## Top 5 Attack (α) Movers

| Team | Frozen α | New α | Change |
|------|----------|-------|--------|
| Norway | 1.023 | 1.149 | +0.126 ↑ |
| Senegal | 0.980 | 1.080 | +0.101 ↑ |
| Netherlands | 1.526 | 1.613 | +0.087 ↑ |
| Canada | 0.850 | 0.937 | +0.086 ↑ |
| Germany | 1.606 | 1.683 | +0.077 ↑ |

## Top 5 Defense (β) Movers

| Team | Frozen β | New β | Change |
|------|----------|-------|--------|
| Iraq | 1.165 | 1.346 | +0.181 ↑ |
| New Zealand | 1.454 | 1.632 | +0.178 ↑ |
| Uzbekistan | 1.254 | 1.427 | +0.173 ↑ |
| Tunisia | 0.878 | 1.031 | +0.153 ↑ |
| Cabo Verde | 1.153 | 1.018 | -0.136 ↓ |

## Prediction Flips (0 of 16 matches changed)

*No prediction flips — both models agree on all 16 Round of 32 outcomes.*

## Full Prediction Comparison

| Match | Home | Away | Frozen P(H)/P(D)/P(A) | Winner (frozen) | New P(H)/P(D)/P(A) | Winner (new) | Flip? |
|-------|------|------|-----------------------|-----------------|---------------------|--------------|-------|
| 73 | Germany | Paraguay | 0.596/0.215/0.189 | **Germany** | 0.622/0.206/0.173 | **Germany** |  |
| 74 | France | Sweden | 0.652/0.203/0.144 | **France** | 0.668/0.192/0.140 | **France** |  |
| 75 | South Africa | Canada | 0.303/0.310/0.386 | **Canada** | 0.276/0.303/0.420 | **Canada** |  |
| 76 | Netherlands | Morocco | 0.411/0.279/0.310 | **Netherlands** | 0.424/0.269/0.307 | **Netherlands** |  |
| 77 | Portugal | Croatia | 0.523/0.240/0.236 | **Portugal** | 0.540/0.237/0.223 | **Portugal** |  |
| 78 | Spain | Austria | 0.500/0.244/0.256 | **Spain** | 0.526/0.239/0.235 | **Spain** |  |
| 79 | United States | Bosnia and Herzegovina | 0.458/0.274/0.268 | **United States** | 0.481/0.258/0.261 | **United States** |  |
| 80 | Belgium | Senegal | 0.534/0.247/0.219 | **Belgium** | 0.539/0.237/0.224 | **Belgium** |  |
| 81 | Brazil | Japan | 0.621/0.205/0.174 | **Brazil** | 0.619/0.206/0.175 | **Brazil** |  |
| 82 | Côte d'Ivoire | Norway | 0.440/0.258/0.303 | **Côte d'Ivoire** | 0.437/0.252/0.311 | **Côte d'Ivoire** |  |
| 83 | Mexico | Ecuador | 0.364/0.279/0.357 | **Mexico** | 0.381/0.287/0.331 | **Mexico** |  |
| 84 | England | DR Congo | 0.694/0.189/0.117 | **England** | 0.664/0.205/0.131 | **England** |  |
| 85 | Argentina | Cabo Verde | 0.708/0.194/0.098 | **Argentina** | 0.682/0.213/0.105 | **Argentina** |  |
| 86 | Australia | Egypt | 0.343/0.291/0.366 | **Egypt** | 0.326/0.298/0.377 | **Egypt** |  |
| 87 | Switzerland | Algeria | 0.405/0.244/0.351 | **Switzerland** | 0.452/0.234/0.314 | **Switzerland** |  |
| 88 | Colombia | Ghana | 0.585/0.238/0.177 | **Colombia** | 0.568/0.253/0.179 | **Colombia** |  |

---
*Frozen model: predictions based on pre-tournament data only (locked baseline).*
*Re-fitted model: PRIMARY predictions incorporating group stage evidence — use for knockout tracking.*