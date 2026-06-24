# C++ Sinkhorn Router3 Dynamic Scheduling Summary

NP: `6`, fixed `50` iterations, random cost seed `0`, epsilon `0.1`.

| variant | size | count | avg runtime s | first3 s | last3 s | speedup | efficiency | avg obj diff | last suggested weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| contiguous | 8000 | 2 | 4.35416 |  |  | 2.69732 | 0.449554 | 4.26603e-14 | `0.6552446771,1.151594095,1.227116223,0.6336013303,1.146816586,1.185627089` |
| seq | 8000 | 3 | 11.7446 | 11.7446 | 11.7446 |  |  | 0 | `` |
| weighted_adaptive | 8000 | 2 | 4.38154 |  |  | 2.68047 | 0.446744 | 3.81431e-14 | `0.547121699,1.071014617,1.387109157,0.5638261018,1.085703067,1.345225359` |
| weighted_equal | 8000 | 2 | 4.36659 |  |  | 2.68964 | 0.448274 | 3.88717e-14 | `0.5769103886,1.101179975,1.30046643,0.60063085,1.101650189,1.319162167` |
| weighted_manual_211 | 8000 | 2 | 5.84768 |  |  | 2.00842 | 0.334736 | 3.72202e-14 | `0.5748273683,1.083187796,1.370292712,0.5731749827,1.068950026,1.329567115` |

Adaptive weights used per rep are in `adaptive_weights_by_rep.csv`.
Raw per-run data is in `per_run.csv`; logs are in `logs/`.
