# C++ Sinkhorn Router Scaling Comparison, fixed 50 iterations

Main comparison uses matching config: random cost seed `0`, epsilon `0.1`, fixed `50` iterations, sizes `2000-12000`, `4` reps.

- `router2`: `192.168.1.25 + 192.168.1.26`, MPI `np=2`, router Ethernet path.
- `router3`: `192.168.1.25 + 192.168.1.26 + 192.168.1.75`, MPI `np=3`, router Ethernet path.
- `vm4`: historical 4-node run `.105/.227/.159/.190`, MPI `np=4`, same solver/config; included as broader scaling context.

Older two-node same-band Wi-Fi variant results used epsilon `0.5`, so they are not mixed into this apples-to-apples table.

| size | seq avg4 s | router2 np2 s | router2 speedup | router3 np3 s | router3 speedup | router3 vs router2 runtime | vm4 np4 s | vm4 speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.568318 | 0.451849 | 1.2578x | 0.557330 | 1.1429x | 0.8107x | 1.084032 | 0.5876x |
| 3000 | 1.300464 | 0.894609 | 1.4537x | 0.759773 | 1.7557x | 1.1775x | 1.170795 | 1.1393x |
| 4000 | 2.203882 | 1.507467 | 1.4620x | 1.132659 | 2.1817x | 1.3309x | 1.528338 | 1.6169x |
| 5000 | 3.602922 | 2.093262 | 1.7212x | 1.700114 | 2.0895x | 1.2312x | 1.845597 | 1.9248x |
| 6000 | 4.979196 | 3.007993 | 1.6553x | 2.213998 | 2.4393x | 1.3586x | 2.473191 | 2.1837x |
| 7000 | 6.820128 | 3.634253 | 1.8766x | 2.694051 | 2.5110x | 1.3490x | 2.934456 | 2.3053x |
| 8000 | 8.987481 | 4.708053 | 1.9090x | 3.677253 | 2.4152x | 1.2803x | 3.450585 | 2.5739x |
| 9000 | 11.518247 | 5.786263 | 1.9906x | 4.262494 | 2.6335x | 1.3575x | 4.196873 | 2.6746x |
| 10000 | 13.499584 | 7.028811 | 1.9206x | 4.994542 | 2.6852x | 1.4073x | 4.659614 | 2.8782x |
| 11000 | 17.217508 | 9.079429 | 1.8963x | 5.941507 | 2.6220x | 1.5281x | 5.783840 | 2.6935x |
| 12000 | 19.602529 | 10.735582 | 1.8259x | 6.816876 | 2.8859x | 1.5749x | 6.142765 | 3.2026x |

## Size 12000 snapshot

At size `12000`, router2 reached `1.8259x` speedup with `10.735582s` MPI runtime.
Router3 reached `2.8859x` and was `1.5749x` faster than router2 by MPI runtime.
The historical VM4 run reached `3.2026x` and was `1.7477x` faster than router2 by MPI runtime.

Per-run details remain in each benchmark directory: `per_run.csv`, `summary.csv`, and `summary.md`.
