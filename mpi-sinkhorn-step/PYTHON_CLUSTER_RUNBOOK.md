# Python MPI Cluster Runbook

This branch is the Python MPI MVP for the Sinkhorn optimal transport project.
Use this guide to set up the cluster, connect machines, install dependencies,
run smoke tests, and reproduce a basic demo.

## Branch Purpose

Branch: `main`

Main contents:

- `src/mpi_sinkhorn/`: Python Sinkhorn solver implementation.
- `apps/sequential_sinkhorn_runner.py`: sequential benchmark runner.
- `apps/mpi_sinkhorn_runner.py`: MPI benchmark runner.
- `apps/hello_mpi.py`: simple cluster connectivity test.
- `apps/compare_sinkhorn_results.py`: compares sequential and MPI JSON output.

The MPI version partitions the cost matrix by rows. Each rank computes local
matrix-vector work, then uses `MPI.Allreduce` for the global column update and
for convergence metrics.

## Recommended Machine Setup

Use Ubuntu Server VMs with bridged networking. This is the most reliable setup
for OpenMPI multi-node runs in this project.

Recommended per physical machine:

1. Install VirtualBox.
2. Create one Ubuntu Server VM.
3. Set the VM network adapter to `Bridged Adapter`.
4. Connect all physical machines to the same LAN or phone hotspot.
5. Use one VM per physical machine.

Avoid WSL for the final demo. WSL can work for local development, but it caused
unstable OpenMPI behavior and difficult networking for multi-node runs.

## Install Dependencies On Every VM

Run on every Ubuntu VM:

```bash
sudo apt update
sudo apt install -y openssh-server openmpi-bin libopenmpi-dev \
  python3-mpi4py python3-venv python3-pip git rsync
sudo systemctl enable --now ssh
```

Check that local MPI works before trying the cluster:

```bash
mpirun -np 1 hostname
python3 -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())'
```

Expected output:

```text
<local-hostname>
0 1
```

## Configure SSH From Master To Workers

On the master VM:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_mpi -N ""
ssh-copy-id -i ~/.ssh/id_ed25519_mpi.pub <user>@<worker-ip>
ssh -i ~/.ssh/id_ed25519_mpi <user>@<worker-ip> hostname
```

Repeat `ssh-copy-id` for every worker.

For the two-VM benchmark environment used during development:

```text
master VM: 192.168.1.105
worker VM: 192.168.1.227
```

Your IP addresses may be different. Use `hostname -I` inside each VM to check.

## Create Hostfile

On the master VM:

```bash
cat > ~/hosts <<'EOF'
192.168.1.105 slots=2
192.168.1.227 slots=2
EOF
```

For three or four physical machines, add one line per VM:

```text
<master-ip> slots=2
<worker-1-ip> slots=2
<worker-2-ip> slots=2
<worker-3-ip> slots=2
```

Use `slots` to control how many MPI processes can run on each node.

## Clone The Repository On Every VM

Every node should have the project at the same path:

```bash
cd ~
git clone https://github.com/sontungkieu/Parallel-Programming-Project.git
cd Parallel-Programming-Project
git checkout main
```

If a worker cannot access GitHub, sync from master:

```bash
rsync -av ~/Parallel-Programming-Project/ \
  <user>@<worker-ip>:~/Parallel-Programming-Project/ \
  -e "ssh -i /home/<user>/.ssh/id_ed25519_mpi"
```

## Install The Python Package

Run this on every VM:

```bash
cd ~/Parallel-Programming-Project/mpi-sinkhorn-step
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If the worker is slow during `pip install`, wait until it finishes before
running MPI. Every node must have the same virtual environment path.

## Local Smoke Test

Run on one VM:

```bash
cd ~/Parallel-Programming-Project/mpi-sinkhorn-step
source .venv/bin/activate

python apps/sequential_sinkhorn_runner.py \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/smoke_seq.json

mpirun -np 2 .venv/bin/python apps/mpi_sinkhorn_runner.py \
  --rows 128 \
  --cols 128 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/smoke_mpi_np2.json

python apps/compare_sinkhorn_results.py \
  --sequential results/smoke_seq.json \
  --parallel results/smoke_mpi_np2.json
```

The objective difference should be very small.

## Cluster Connectivity Smoke Test

Run from the master VM:

```bash
cd ~/Parallel-Programming-Project/mpi-sinkhorn-step

mpirun \
  --mca plm_rsh_agent "ssh -i /home/<user>/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=accept-new" \
  --map-by node \
  -np 2 \
  --hostfile ~/hosts \
  --wdir /home/<user>/Parallel-Programming-Project/mpi-sinkhorn-step \
  /home/<user>/Parallel-Programming-Project/mpi-sinkhorn-step/.venv/bin/python \
  apps/hello_mpi.py
```

Expected output should include one line from each rank and hostname.

Use the absolute `.venv/bin/python` path. Remote MPI processes do not inherit
your interactive `source .venv/bin/activate` shell state.

## Cluster Sinkhorn Smoke Test

Run from the master VM:

```bash
cd ~/Parallel-Programming-Project/mpi-sinkhorn-step

python apps/sequential_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/demo_seq_1000.json

mpirun \
  --mca plm_rsh_agent "ssh -i /home/<user>/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=accept-new" \
  --map-by node \
  -np 2 \
  --hostfile ~/hosts \
  --wdir /home/<user>/Parallel-Programming-Project/mpi-sinkhorn-step \
  /home/<user>/Parallel-Programming-Project/mpi-sinkhorn-step/.venv/bin/python \
  apps/mpi_sinkhorn_runner.py \
  --rows 1000 \
  --cols 1000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/demo_mpi_1000_np2.json

python apps/compare_sinkhorn_results.py \
  --sequential results/demo_seq_1000.json \
  --parallel results/demo_mpi_1000_np2.json
```

## Scaling To More Machines

For 3 machines with 2 slots each:

```bash
mpirun \
  --mca plm_rsh_agent "ssh -i /home/<user>/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=accept-new" \
  --map-by node \
  -np 6 \
  --hostfile ~/hosts \
  --wdir /home/<user>/Parallel-Programming-Project/mpi-sinkhorn-step \
  /home/<user>/Parallel-Programming-Project/mpi-sinkhorn-step/.venv/bin/python \
  apps/mpi_sinkhorn_runner.py \
  --rows 2000 \
  --cols 2000 \
  --cost-mode random \
  --seed 0 \
  --epsilon 0.5 \
  --max-iters 50 \
  --tol 1e-6 \
  --output results/demo_mpi_2000_np6.json
```

Always verify local MPI, SSH, and `apps/hello_mpi.py` before running the solver.

## Output Files

Both sequential and MPI runners write JSON with:

- `runtime_sec`
- `iterations`
- `row_error`
- `col_error`
- `transport_objective`
- `num_processes`
- `hostnames`

Use `apps/compare_sinkhorn_results.py` to compute speedup and verify numerical
agreement between sequential and MPI runs.

## Troubleshooting

If `mpirun -np 1 hostname` fails locally, fix OpenMPI on that node first:

```bash
sudo apt update
sudo apt install --reinstall -y openmpi-bin libopenmpi-dev python3-mpi4py
```

If SSH works but MPI hangs:

- Confirm all VMs are on the same LAN or hotspot.
- Confirm VirtualBox uses bridged networking.
- Confirm no node is using WSL for the cluster demo.
- Confirm the repository path is identical on every node.
- Confirm the virtual environment exists on every node.
- Run `mpirun ... apps/hello_mpi.py` before the Sinkhorn solver.
