| Arm | Accuracy | Macro F1 | Avg latency (s) | n |
| --- | --- | --- | --- | --- |
| plain | 30.00% | 29.69% | 6.4 | 200 |
| plain_rr | 37.00% | 35.21% | 6.6 | 200 |
| graph | 59.50% | 50.51% | 7.5 | 200 |
| graph_concepts | 57.50% | 49.97% | 40.8 | 200 |

### Significance (paired McNemar)

| Contrast | Δacc (pp) | gains | losses | p | sig? |
| --- | --- | --- | --- | --- | --- |
| plain → plain_rr (reranker effect) | +7.00 | 35 | 21 | 0.0814 | no |
| plain_rr → graph (parent-expansion effect) | +22.50 | 71 | 26 | 0.0000 | yes |
| graph → graph_concepts (concept-hop effect) | -2.00 | 26 | 30 | 0.6889 | no |
