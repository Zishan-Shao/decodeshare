# A1 Cross-method candidate pools: RankFlip summary

This table summarizes protocol-level ranking generalization under KV-cached decode.

Metrics are computed per pool using mean scores across template seeds.

| Pool | #Vec | Spearman(TRAD,REAL) | Spearman(DECODE,REAL) | Spearman(TRAD,DECODE) | regret@1 TRAD | regret@1 DECODE | top10 overlap TRAD∩REAL | top10 overlap DECODE∩REAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CAA | 8 | 0.455 | 0.563 | 0.163 | 0.0047 | 0.0000 | 8 | 8 |
| INSTR | 64 | 0.172 | 0.767 | 0.072 | 0.0156 | 0.0031 | 2 | 2 |
| SAE | 64 | -0.064 | 0.594 | -0.134 | 0.0125 | 0.0031 | 1 | 6 |
