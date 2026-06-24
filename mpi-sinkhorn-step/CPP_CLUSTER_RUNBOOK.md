# C++ MPI Cluster Runbook

This branch contains the C++ implementation and benchmark results for the
MPI Sinkhorn project. Use this guide when setting up the cluster, building the
C++ binary, running smoke tests, and reproducing the benchmark charts.

## Branch Purpose

Branch: `codex/cpp-benchmark`

Main additions:

- `mpi-sinkhorn-step/cpp/sinkhorn_cpp.cpp`: C++17 dense Sinkhorn solver.
- `mpi-sinkhorn-step/cpp/Makefile`: builds the C++ MPI binary with `mpicxx`.
- `mpi-sinkhorn-step/results/bench_cpp_vm2_sameband_variants_fixed50/`: final
  benchmark tables, findings, and charts.

The C++ MPI implementation partitions the cost matrix by rows. Each rank owns a
block of rows, computes local `u` updates and local column sums, then synchronizes
the global column vector used for `v`.

## Recommended Machine Setup

Use Ubuntu Server VMs with a bridged network adapter. This avoids WSL NAT,
mirrored networking, and Windows Hyper-V firewall issues.

For each physical machine:

1. Install VirtualBox.
2. Create one Ubuntu Server VM.
3. Set VM network mode to `Bridged Adapter`.
4. Connect all physical machines to the same LAN or phone hotspot.
5. Give each VM enough resources:
   - 4 vCPU if available.
   - 4 GB RAM minimum for the tested sizes.
   - 20 GB disk is enough for code and results.

The benchmark in this branch was run with two Ubuntu VMs:

```text
master VM: 192.168.1.105
worker VM: 192.168.1.227
```

You can replace these addresses with your own VM IPs.

## Install Dependencies On Every VM

Run this on each VM:

```bash
sudo apt update
sudo apt install -y openssh-server openmpi-bin libopenmpi-dev \
  python3 python3-venv python3-pip git rsync build-essential
sudo systemctl enable --now ssh
```

Check local MPI before trying multiple machines:

```bash
mpirun -np 1 hostname
```

Expected result: the local hostname prints and the command exits cleanly.

## Configure SSH From Master To Workers

On the master VM:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_mpi -N ""
ssh-copy-id -i ~/.ssh/id_ed25519_mpi.pub tung@192.168.1.227
ssh -i ~/.ssh/id_ed25519_mpi tung@192.168.1.227 hostname
```

Replace `tung` and `192.168.1.227` with your VM user and worker IP.

If you use more workers, repeat `ssh-copy-id` for every worker.

## Create Hostfile

On the master VM:

```bash
cat > ~/hosts <<'EOF'
192.168.1.105 slots=2
192.168.1.227 slots=2
EOF
```

For a 3-machine or 4-machine demo, add more lines:

```text
<master-ip> slots=2
<worker-1-ip> slots=2
<worker-2-ip> slots=2
<worker-3-ip> slots=2
```

Use the number of slots that matches the CPU cores you want MPI to use.

## Clone Or Sync The Repository

On every VM, the project should live at the same path:

```bash
cd ~
git clone https://github.com/sontungkieu/Parallel-Programming-Project.git
cd Parallel-Programming-Project
git checkout codex/cpp-benchmark
```

If SSH to GitHub is configured, you can use:

```bash
git clone git@github.com:sontungkieu/Parallel-Programming-Project.git
```

If the worker has no GitHub access, sync from master:

```bash
rsync -av ~/Parallel-Programming-Project/ \
  tung@192.168.1.227:~/Parallel-Programming-Project/ \
  -e "ssh -i /home/tung/.ssh/id_ed25519_mpi"
```

## Build The C++ Binary

On every VM:

```bash
cd ~/Parallel-Programming-Project/mpi-sinkhorn-step/cpp
make clean all
```

The binary should be created at:

```text
mpi-sinkhorn-step/cpp/sinkhorn_cpp
```

## Local Smoke Test

Run on one VM:

```bash
cd ~/Parallel-Programming-Project/mpi-sinkhorn-step

cpp/sinkhorn_cpp \
  --mode sequential \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 0 \
  --check-every 50 \
  --output results/cpp_local_seq_128.json

mpirun -np 2 cpp/sinkhorn_cpp \
  --mode mpi \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 0 \
  --check-every 50 \
  --output results/cpp_local_mpi_128.json
```

Compare with the existing Python comparison helper:

```bash
python3 apps/compare_sinkhorn_results.py \
  --sequential results/cpp_local_seq_128.json \
  --parallel results/cpp_local_mpi_128.json
```

## Two-Node Cluster Smoke Test

Run from the master VM:

```bash
cd ~/Parallel-Programming-Project/mpi-sinkhorn-step

mpirun \
  --mca plm_rsh_agent "ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=accept-new" \
  --map-by node \
  -np 2 \
  --hostfile ~/hosts \
  --wdir /home/tung/Parallel-Programming-Project/mpi-sinkhorn-step \
  cpp/sinkhorn_cpp \
  --mode mpi \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 0 \
  --check-every 50 \
  --output results/cpp_cluster_mpi_128.json
```

Expected output includes both hostnames and:

```text
C++ MPI Sinkhorn complete
processes=2
```

## Communication Variants

The C++ program supports these MPI communication variants:

```text
--comm-mode double
--comm-mode float32
--comm-mode quantized
--comm-mode lazy --sync-every 5
--kernel-cutoff 0.25
```

Meaning:

- `double`: baseline double precision `MPI_Allreduce`.
- `float32`: casts the column vector to `float` before `MPI_Allreduce`.
- `quantized`: sends fixed-point `uint32_t` values.
- `lazy --sync-every 5`: synchronizes the column vector every 5 iterations.
- `--kernel-cutoff 0.25`: zeroes small kernel values. The current code still
  scans dense storage, so this is not a good runtime optimization yet.

## Reproduce A Single Large Run

Example at size 12000 with fixed 50 iterations:

```bash
cd ~/Parallel-Programming-Project/mpi-sinkhorn-step

cpp/sinkhorn_cpp \
  --mode sequential \
  --rows 12000 \
  --cols 12000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 0 \
  --check-every 50 \
  --output results/manual_seq_12000.json

mpirun \
  --mca plm_rsh_agent "ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=accept-new" \
  --map-by node \
  -np 2 \
  --hostfile ~/hosts \
  --wdir /home/tung/Parallel-Programming-Project/mpi-sinkhorn-step \
  cpp/sinkhorn_cpp \
  --mode mpi \
  --comm-mode lazy \
  --sync-every 5 \
  --rows 12000 \
  --cols 12000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 0 \
  --check-every 50 \
  --output results/manual_mpi_lazy_12000.json
```

## Read Final Results

The final benchmark summary is here:

```text
results/bench_cpp_vm2_sameband_variants_fixed50/findings.md
```

Useful chart files:

```text
results/bench_cpp_vm2_sameband_variants_fixed50/charts/practical_variants_speedup_runtime.png
results/bench_cpp_vm2_sameband_variants_fixed50/charts/speedup_by_size.png
results/bench_cpp_vm2_sameband_variants_fixed50/charts/size12000_runtime_speedup.png
```

The best practical result in the recorded benchmark:

```text
N = 12000
lazy sync=5 runtime = 9.142226 sec
lazy sync=5 speedup = 1.8915x
```

## Troubleshooting

If `mpirun -np 1 hostname` fails locally, reinstall OpenMPI before debugging the
network:

```bash
sudo apt update
sudo apt install --reinstall -y openmpi-bin libopenmpi-dev
```

If SSH works but MPI hangs:

- Confirm every VM is on the same LAN or hotspot.
- Confirm the VMs use bridged networking.
- Confirm every VM has the same project path.
- Confirm the C++ binary exists on every VM.
- Try `ping <worker-ip>` from master.
- Try `ssh -i ~/.ssh/id_ed25519_mpi <user>@<worker-ip> hostname`.

For this project, avoid WSL for the final cluster demo. WSL networking was less
reliable than bridged Ubuntu VMs for OpenMPI multi-node runs.
