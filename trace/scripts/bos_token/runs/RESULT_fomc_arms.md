# RESULT: BoS-token vs anchor-generated C-STANCE replay -> FOMC round

Round-0 C-STANCE reference (1phase_kd_rep, real replay): 58.35

| arm | C-STANCE@r1 | FOMC@r1 | C-STANCE forgetting (58.35 - score) |
|---|---|---|---|
| bos_gen | 56.85 | 73.99 | 1.50 |
| anchor_gen | 57.55 | 71.37 | 0.80 |
| real_replay (ref) | 57.85 | 71.17 | 0.50 |

## bos_gen replay selection stats

```
{
 "total": 768,
 "passed": 755,
 "answer_hist": {
  "A": 245,
  "B": 274,
  "C": 236
 },
 "mean_body_len": 112.61986754966887,
 "selected": 500
}
```
