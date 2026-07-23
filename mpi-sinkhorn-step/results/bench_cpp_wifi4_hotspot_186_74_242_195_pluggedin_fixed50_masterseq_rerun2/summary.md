# Rerun 2: 4-machine hotspot fixed-50 plus master sequential

- Hosts: `10.110.77.186`, `10.110.77.74`, `10.110.77.242`, `10.110.77.195`
- Workload: `N=12000`, `max-iters=50`, `seed=0`, `epsilon=0.1`, `comm-mode=double`

| kind | rep | runtime_sec | wall_clock_sec | speedup_vs_seq_avg | objective |
|---|---:|---:|---:|---:|---:|
| seq_master | 1 | 29.867896 | 32.31 | 1.000 | 0.099999182361 |
| seq_master | 2 | 30.343815 | 32.77 | 1.000 | 0.099999182361 |
| seq_master | 3 | 29.133498 | 31.28 | 1.000 | 0.099999182361 |
| mpi4 | 1 | 13.375273 | 33.73 | 2.227 | 0.099999182361 |
| mpi4 | 2 | 13.554048 | 15.82 | 2.197 | 0.099999182361 |
| mpi4 | 3 | 13.361064 | 16.66 | 2.229 | 0.099999182361 |

Aggregate:
- master sequential avg3 = `29.781737 s`
- MPI 4-machine avg3 = `13.430128 s`
- speedup vs current master avg3 = `2.218x`
- efficiency vs 4 processes = `0.554`
- MPI min/max = `13.361064 / 13.554048 s`

MPI suggested weights by rep:
- rep 1: `0.5492825335,0.9709825196,1.147115206,1.332619741`
- rep 2: `0.5598604469,0.9716517723,1.148968447,1.319519334`
- rep 3: `0.5564980084,0.9896336946,1.11813274,1.335735557`

Ping from master before run:
```text
timestamp=2026-07-03T07:58:35+00:00
master=vmubuntu 10.110.77.186 2401:d800:2cd:acef:a00:27ff:fe2b:61ff 
--- ping 10.110.77.74 ---
PING 10.110.77.74 (10.110.77.74) 56(84) bytes of data.
64 bytes from 10.110.77.74: icmp_seq=1 ttl=64 time=13.9 ms
64 bytes from 10.110.77.74: icmp_seq=2 ttl=64 time=13.3 ms
64 bytes from 10.110.77.74: icmp_seq=3 ttl=64 time=10.5 ms

--- 10.110.77.74 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 5084ms
rtt min/avg/max/mdev = 10.503/12.567/13.933/1.484 ms
--- ping 10.110.77.242 ---
PING 10.110.77.242 (10.110.77.242) 56(84) bytes of data.
64 bytes from 10.110.77.242: icmp_seq=1 ttl=64 time=13.3 ms
64 bytes from 10.110.77.242: icmp_seq=2 ttl=64 time=11.5 ms
64 bytes from 10.110.77.242: icmp_seq=3 ttl=64 time=18.2 ms

--- 10.110.77.242 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 3690ms
rtt min/avg/max/mdev = 11.481/14.333/18.205/2.838 ms
--- ping 10.110.77.195 ---
PING 10.110.77.195 (10.110.77.195) 56(84) bytes of data.
64 bytes from 10.110.77.195: icmp_seq=1 ttl=64 time=12.6 ms
64 bytes from 10.110.77.195: icmp_seq=2 ttl=64 time=19.7 ms
64 bytes from 10.110.77.195: icmp_seq=3 ttl=64 time=13.7 ms

--- 10.110.77.195 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 4410ms
rtt min/avg/max/mdev = 12.649/15.358/19.708/3.106 ms
```
