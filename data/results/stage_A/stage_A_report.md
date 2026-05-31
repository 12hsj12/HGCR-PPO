# Stage A Report

## Baseline Summary

| size | method | Cmax mean | Cmax std | valid ratio |
|---|---:|---:|---:|---:|
| large | FIFO | 525.818 | 22.855 | 1.00 |
| large | GreedyECT | 545.151 | 26.520 | 1.00 |
| large | LPT | 637.595 | 33.179 | 1.00 |
| large | MinCandidateLoad | 591.521 | 41.053 | 1.00 |
| large | Random | 578.431 | 44.189 | 1.00 |
| large | SPT | 700.997 | 62.408 | 1.00 |
| medium | FIFO | 334.543 | 32.238 | 1.00 |
| medium | GreedyECT | 361.560 | 32.500 | 1.00 |
| medium | LPT | 403.616 | 28.997 | 1.00 |
| medium | MinCandidateLoad | 373.000 | 31.975 | 1.00 |
| medium | Random | 391.649 | 29.112 | 1.00 |
| medium | SPT | 440.972 | 45.323 | 1.00 |
| small | FIFO | 220.161 | 27.904 | 1.00 |
| small | GreedyECT | 230.943 | 28.902 | 1.00 |
| small | LPT | 242.748 | 26.392 | 1.00 |
| small | MinCandidateLoad | 223.707 | 27.873 | 1.00 |
| small | Random | 241.436 | 29.186 | 1.00 |
| small | SPT | 259.782 | 34.085 | 1.00 |

## Best Heuristic By Size

| size | best method | Cmax mean |
|---|---:|---:|
| large | FIFO | 525.818 |
| medium | FIFO | 334.543 |
| small | FIFO | 220.161 |

## Split Effect Summary

| size | ordering | split strategy | Cmax mean | valid ratio | cmax check pass ratio |
|---|---|---|---:|---:|---:|
| large | FIFO | EqualSplit | 526.183 | 1.00 | 1.00 |
| large | FIFO | GreedyECTSplit | 525.818 | 1.00 | 1.00 |
| large | FIFO | MaxSplit | 525.818 | 1.00 | 1.00 |
| large | FIFO | NoSplit | 532.602 | 1.00 | 1.00 |
| large | FIFO | OracleSplitDebug | 524.988 | 1.00 | 1.00 |
| large | FIFO | RandomSplit | 531.835 | 1.00 | 1.00 |
| large | FIFO | SpeedRatioSplit | 525.818 | 1.00 | 1.00 |
| large | GreedyECT | EqualSplit | 547.112 | 1.00 | 1.00 |
| large | GreedyECT | GreedyECTSplit | 545.151 | 1.00 | 1.00 |
| large | GreedyECT | MaxSplit | 545.151 | 1.00 | 1.00 |
| large | GreedyECT | NoSplit | 546.119 | 1.00 | 1.00 |
| large | GreedyECT | OracleSplitDebug | 544.176 | 1.00 | 1.00 |
| large | GreedyECT | RandomSplit | 545.314 | 1.00 | 1.00 |
| large | GreedyECT | SpeedRatioSplit | 545.151 | 1.00 | 1.00 |
| medium | FIFO | EqualSplit | 336.786 | 1.00 | 1.00 |
| medium | FIFO | GreedyECTSplit | 334.543 | 1.00 | 1.00 |
| medium | FIFO | MaxSplit | 334.543 | 1.00 | 1.00 |
| medium | FIFO | NoSplit | 336.731 | 1.00 | 1.00 |
| medium | FIFO | OracleSplitDebug | 330.438 | 1.00 | 1.00 |
| medium | FIFO | RandomSplit | 333.744 | 1.00 | 1.00 |
| medium | FIFO | SpeedRatioSplit | 334.543 | 1.00 | 1.00 |
| medium | GreedyECT | EqualSplit | 360.472 | 1.00 | 1.00 |
| medium | GreedyECT | GreedyECTSplit | 361.560 | 1.00 | 1.00 |
| medium | GreedyECT | MaxSplit | 361.560 | 1.00 | 1.00 |
| medium | GreedyECT | NoSplit | 357.667 | 1.00 | 1.00 |
| medium | GreedyECT | OracleSplitDebug | 362.864 | 1.00 | 1.00 |
| medium | GreedyECT | RandomSplit | 362.376 | 1.00 | 1.00 |
| medium | GreedyECT | SpeedRatioSplit | 361.560 | 1.00 | 1.00 |
| small | FIFO | EqualSplit | 220.836 | 1.00 | 1.00 |
| small | FIFO | GreedyECTSplit | 220.161 | 1.00 | 1.00 |
| small | FIFO | MaxSplit | 220.161 | 1.00 | 1.00 |
| small | FIFO | NoSplit | 221.676 | 1.00 | 1.00 |
| small | FIFO | OracleSplitDebug | 216.318 | 1.00 | 1.00 |
| small | FIFO | RandomSplit | 221.440 | 1.00 | 1.00 |
| small | FIFO | SpeedRatioSplit | 220.161 | 1.00 | 1.00 |
| small | GreedyECT | EqualSplit | 231.656 | 1.00 | 1.00 |
| small | GreedyECT | GreedyECTSplit | 230.943 | 1.00 | 1.00 |
| small | GreedyECT | MaxSplit | 230.943 | 1.00 | 1.00 |
| small | GreedyECT | NoSplit | 233.097 | 1.00 | 1.00 |
| small | GreedyECT | OracleSplitDebug | 231.010 | 1.00 | 1.00 |
| small | GreedyECT | RandomSplit | 234.285 | 1.00 | 1.00 |
| small | GreedyECT | SpeedRatioSplit | 230.943 | 1.00 | 1.00 |

## Schedule Validation

- Baseline groups with invalid schedules: 0
- Split-effect groups with invalid schedules: 0
- Any invalid schedule detected: no

## Next Stage Recommendation

Stage A has fixed datasets, unified evaluation, split-effect diagnostics, and schedule validation in place. The next step is Stage B: build LookaheadGreedy, BeamSearch, and HybridTopK candidate sets for later candidate-set experiments.
