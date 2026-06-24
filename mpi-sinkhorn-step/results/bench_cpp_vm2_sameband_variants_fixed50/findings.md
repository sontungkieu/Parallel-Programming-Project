# C++ Communication Variant Findings

Environment: two Ubuntu VMs on the same Wi-Fi band, OpenMPI, `np=2`, fixed 50 iterations, 3-run averages. Ping before benchmark: 0% packet loss, RTT avg 5.486 ms.

## Variants Tested

| Variant | Idea | Main column payload | Notes |
|---|---|---:|---|
| double baseline | Original MPI double `Allreduce` every iteration | 1.000x | Reference implementation |
| float32 comm | Cast column vector to `float` for `Allreduce` | 0.500x | Lower payload, small precision loss |
| quantized comm | Fixed-point `uint32_t` column vector for `Allreduce` | 0.500x | Lower payload, extra pack/unpack overhead |
| lazy sync=5 | Sync column vector every 5 iterations | 0.200x | Best runtime in this benchmark |
| sparse cutoff 0.25 | Zero kernel values below cutoff | 1.000x | Changes objective; dense masked storage is slow |

## Best Large-Size Results

| size | double baseline MPI | float32 MPI | quantized MPI | lazy sync=5 MPI | best variant |
|---:|---:|---:|---:|---:|---|
| 10000 | 8.228874s | 7.175473s | 7.175616s | 6.440461s | lazy sync=5 |
| 11000 | 9.653485s | 8.615356s | 8.520366s | 7.611377s | lazy sync=5 |
| 12000 | 11.195551s | 9.895247s | 9.997453s | 9.142226s | lazy sync=5 |

## Speedup At Size 12000

| Variant | Speedup vs matching sequential | MPI runtime | Max objective diff |
|---|---:|---:|---:|
| double baseline | 1.5446x | 11.195551s | 8.865e-14 |
| float32 comm | 1.7476x | 9.895247s | 5.532e-11 |
| quantized comm | 1.7297x | 9.997453s | 1.351e-09 |
| lazy sync=5 | 1.8915x | 9.142226s | 9.071e-14 |
| sparse cutoff 0.25 | 1.8614x | 29.595221s | 7.655e-14 vs sparse sequential |

## Takeaways

Lazy sync every 5 iterations is the strongest practical optimization here. It reduces main column communication to 20% of baseline and gives the best large-size speedup, reaching 1.8915x at size 12000.

Float32 communication is the safest compression-style change. It halves payload and improves runtime at large sizes while keeping objective differences around 1e-10 to 1e-11.

Quantized communication also halves payload, but pack/unpack overhead and higher numerical error make it less attractive than float32 in this setup.

Sparse cutoff should not be presented as a runtime optimization in the current implementation. It changes the objective and is much slower in absolute runtime because the code still stores and scans a dense matrix. It would need a real sparse representation to become a fair optimization.
