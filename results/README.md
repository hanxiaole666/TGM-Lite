# Results archive

Exact CSV outputs behind every table in the paper. `*_5seeds.csv` = aggregate
(mean ± std over seeds 42–46); `*_5seeds_raw.csv` = per-seed rows (each row also
carries parameter count, seconds/epoch, peak GPU memory, inference latency, and
the three drift-block F1 scores w1/w2/w3).

| File | Paper table | Run command (see repo README) |
|---|---|---|
| `amlsim_time_extrapolation_5seeds.csv` | Table II | AMLSim, time extrapolation |
| `amlsim_cross_bank_5seeds.csv` | Table III | AMLSim, 30 % held-out banks |
| `amlsim_label_scarcity_5pct_5seeds.csv` | Table V | AMLSim, 5 % training labels (RF/MLP/GCN/TGM-Lite) |
| `amlsim_label_scarcity_5pct_evolvegcn_5seeds.csv` | Table V | AMLSim, 5 % training labels (EvolveGCN) |
| `elliptic_time_extrapolation_5seeds.csv` | Table VII | Elliptic, time extrapolation |
| `elliptic_pu_5seeds.csv` | Table VIII | Elliptic, PU learning (negative result) |

Significance in the paper: Welch t-test on the 5 per-seed values from the raw
files (data split fixed at seed 42; only model initialization varies).
