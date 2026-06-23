#!/usr/bin/env bash
set -u

ROOT="$HOME/Parallel-Programming-Project/mpi-sinkhorn-step"
BIN="$ROOT/cpp/sinkhorn_cpp"
OUT="$ROOT/results/bench_cpp_router2_25_26_fixed50_reps4"
LOG="$OUT/logs"
HOSTFILE="$HOME/hosts_router_2"
SSH_AGENT="ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
MPI_NET="--mca btl tcp,self --mca btl_tcp_if_include enp0s3 --mca oob_tcp_if_include enp0s3"
SIZES="2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000"
REPS=4
NP=2

mkdir -p "$LOG"
make -C "$ROOT/cpp"
ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null tung@192.168.1.26 "mkdir -p '$ROOT/cpp'"
rsync -az -e "$SSH_AGENT" "$BIN" "tung@192.168.1.26:$BIN"

printf "%s\n" "192.168.1.25 slots=2" "192.168.1.26 slots=2" > "$HOSTFILE"
cp "$HOSTFILE" "$OUT/hosts_router_2.txt"
{
  echo "benchmark=bench_cpp_router2_25_26_fixed50_reps4"
  echo "started_at=$(date -Is)"
  echo "root=$ROOT"
  echo "binary=$BIN"
  echo "hostfile=$HOSTFILE"
  echo "hosts=.25,.26"
  echo "sizes=$SIZES"
  echo "reps=$REPS"
  echo "mpi_np=$NP"
  echo "map_by=node"
  echo "network_if=enp0s3"
  echo "max_iters=50"
  echo "epsilon=0.1"
  echo "seed=0"
  echo "cost_mode=random"
} > "$OUT/meta.txt"

echo "SMOKE $(date -Is)" > "$OUT/progress.log"
mpirun -np "$NP" --hostfile "$HOSTFILE" --map-by node $MPI_NET --mca plm_rsh_agent "$SSH_AGENT" hostname </dev/null > "$OUT/mpi_hostname_smoke.log" 2>&1 || {
  echo "smoke_failed" >> "$OUT/failures.txt"
  cat "$OUT/mpi_hostname_smoke.log" >> "$OUT/progress.log"
  exit 1
}

run_seq() {
  local n="$1"
  local rep="$2"
  local json="$OUT/seq_${n}_rep${rep}.json"
  local log="$LOG/seq_${n}_rep${rep}.log"
  local rc=0
  [ -s "$json" ] && { echo "SKIP seq n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"; return 0; }
  echo "RUN seq n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"
  {
    echo "kind=seq"
    echo "size=$n"
    echo "rep=$rep"
    echo "start=$(date -Is)"
    echo "cmd=$BIN --mode sequential --rows $n --cols $n --cost-mode random --seed 0 --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --output $json"
    /usr/bin/time -f "wall_clock_sec=%e max_rss_kb=%M" \
      "$BIN" --mode sequential --rows "$n" --cols "$n" --cost-mode random --seed 0 \
      --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --output "$json"
    rc=$?
    echo "exit_code=$rc"
    echo "end=$(date -Is)"
  } > "$log" 2>&1
  if [ "$rc" -ne 0 ]; then
    echo "FAIL seq n=$n rep=$rep rc=$rc $(date -Is)" | tee -a "$OUT/failures.txt" >> "$OUT/progress.log"
  fi
}

run_mpi() {
  local n="$1"
  local rep="$2"
  local json="$OUT/mpi_${n}_np${NP}_rep${rep}.json"
  local log="$LOG/mpi_${n}_np${NP}_rep${rep}.log"
  local rc=0
  [ -s "$json" ] && { echo "SKIP mpi n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"; return 0; }
  echo "RUN mpi n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"
  {
    echo "kind=mpi"
    echo "size=$n"
    echo "rep=$rep"
    echo "np=$NP"
    echo "start=$(date -Is)"
    echo "cmd=mpirun -np $NP --hostfile $HOSTFILE --map-by node $MPI_NET --mca plm_rsh_agent $SSH_AGENT $BIN --mode mpi --rows $n --cols $n --cost-mode random --seed 0 --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --comm-mode double --output $json"
    /usr/bin/time -f "wall_clock_sec=%e max_rss_kb=%M" timeout -k 20s 300s \
      mpirun -np "$NP" --hostfile "$HOSTFILE" --map-by node $MPI_NET \
      --mca plm_rsh_agent "$SSH_AGENT" \
      "$BIN" --mode mpi --rows "$n" --cols "$n" --cost-mode random --seed 0 \
      --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --comm-mode double --output "$json" </dev/null
    rc=$?
    echo "exit_code=$rc"
    echo "end=$(date -Is)"
  } > "$log" 2>&1
  if [ "$rc" -ne 0 ]; then
    echo "FAIL mpi n=$n rep=$rep rc=$rc $(date -Is)" | tee -a "$OUT/failures.txt" >> "$OUT/progress.log"
  fi
}

for rep in $(seq 1 "$REPS"); do
  for n in $SIZES; do run_seq "$n" "$rep"; done
  for n in $SIZES; do run_mpi "$n" "$rep"; done
done

python3 - "$OUT" "$NP" <<'PY'
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
np = int(sys.argv[2])
pattern = re.compile(r"^(seq|mpi)_(\d+)(?:_np\d+)?_rep(\d+)\.json$")
rows = []

for path in sorted(out.glob("*.json")):
    match = pattern.match(path.name)
    if not match:
        continue
    kind, size, rep = match.group(1), int(match.group(2)), int(match.group(3))
    data = json.loads(path.read_text())
    rows.append({
        "kind": kind,
        "size": size,
        "rep": rep,
        "runtime_sec": float(data["runtime_sec"]),
        "objective": float(data["transport_objective"]),
        "row_error": float(data["row_error"]),
        "col_error": float(data["col_error"]),
        "num_processes": int(data.get("num_processes", 1)),
        "hostnames": ";".join(data.get("hostnames", [])),
        "json": str(path),
        "log": str(out / "logs" / path.name.replace(".json", ".log")),
    })

with (out / "per_run.csv").open("w", newline="") as fh:
    fields = ["kind", "size", "rep", "runtime_sec", "objective", "row_error", "col_error", "num_processes", "hostnames", "json", "log"]
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: (row["size"], row["kind"], row["rep"])))

summary = []
for size in sorted({row["size"] for row in rows}):
    seq = sorted([row for row in rows if row["size"] == size and row["kind"] == "seq"], key=lambda row: row["rep"])
    mpi = sorted([row for row in rows if row["size"] == size and row["kind"] == "mpi"], key=lambda row: row["rep"])
    if not seq or not mpi:
        continue
    seq_by_rep = {row["rep"]: row for row in seq}
    obj_diffs = [abs(row["objective"] - seq_by_rep[row["rep"]]["objective"]) for row in mpi if row["rep"] in seq_by_rep]
    seq_times = [row["runtime_sec"] for row in seq]
    mpi_times = [row["runtime_sec"] for row in mpi]
    seq_avg4 = statistics.mean(seq_times)
    mpi_avg4 = statistics.mean(mpi_times)
    seq_first3 = statistics.mean(seq_times[:3]) if len(seq_times) >= 3 else math.nan
    mpi_first3 = statistics.mean(mpi_times[:3]) if len(mpi_times) >= 3 else math.nan
    seq_last3 = statistics.mean(seq_times[-3:]) if len(seq_times) >= 3 else math.nan
    mpi_last3 = statistics.mean(mpi_times[-3:]) if len(mpi_times) >= 3 else math.nan
    summary.append({
        "size": size,
        "seq_avg4_sec": seq_avg4,
        f"mpi_np{np}_avg4_sec": mpi_avg4,
        "speedup_avg4": seq_avg4 / mpi_avg4,
        "efficiency_avg4": (seq_avg4 / mpi_avg4) / np,
        "seq_first3_sec": seq_first3,
        "mpi_first3_sec": mpi_first3,
        "speedup_first3": seq_first3 / mpi_first3 if mpi_first3 else math.nan,
        "seq_last3_sec": seq_last3,
        "mpi_last3_sec": mpi_last3,
        "speedup_last3": seq_last3 / mpi_last3 if mpi_last3 else math.nan,
        "mpi_min_sec": min(mpi_times),
        "mpi_max_sec": max(mpi_times),
        "max_obj_diff": max(obj_diffs) if obj_diffs else math.nan,
        "max_mpi_row_error": max(row["row_error"] for row in mpi),
        "max_mpi_col_error": max(row["col_error"] for row in mpi),
    })

with (out / "summary.csv").open("w", newline="") as fh:
    fields = ["size", "seq_avg4_sec", f"mpi_np{np}_avg4_sec", "speedup_avg4", "efficiency_avg4", "seq_first3_sec", "mpi_first3_sec", "speedup_first3", "seq_last3_sec", "mpi_last3_sec", "speedup_last3", "mpi_min_sec", "mpi_max_sec", "max_obj_diff", "max_mpi_row_error", "max_mpi_col_error"]
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

def fmt(value, digits=6):
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)

lines = [
    "# C++ Sinkhorn Router2 fixed50 reps4 summary",
    "",
    "Cluster: `192.168.1.25`, `192.168.1.26`; MPI `np=2`, `--map-by node`, OpenMPI forced to `enp0s3`.",
    "",
    "Config: random cost seed `0`, epsilon `0.1`, fixed `50` iterations, sizes `2000-12000`, `4` reps.",
    "",
    "## Main Averages",
    "",
    "| size | seq avg4 s | mpi np2 avg4 s | speedup avg4 | eff avg4 | seq last3 s | mpi last3 s | speedup last3 | mpi min-max s | max obj diff |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in summary:
    lines.append(
        f"| {row['size']} | {fmt(row['seq_avg4_sec'])} | {fmt(row[f'mpi_np{np}_avg4_sec'])} | {row['speedup_avg4']:.4f}x | {row['efficiency_avg4']:.4f} | {fmt(row['seq_last3_sec'])} | {fmt(row['mpi_last3_sec'])} | {row['speedup_last3']:.4f}x | {fmt(row['mpi_min_sec'])}-{fmt(row['mpi_max_sec'])} | {row['max_obj_diff']:.3e} |"
    )
lines += [
    "",
    "## First3 vs Last3",
    "",
    "| size | seq first3 s | mpi first3 s | speedup first3 | seq last3 s | mpi last3 s | speedup last3 |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in summary:
    lines.append(
        f"| {row['size']} | {fmt(row['seq_first3_sec'])} | {fmt(row['mpi_first3_sec'])} | {row['speedup_first3']:.4f}x | {fmt(row['seq_last3_sec'])} | {fmt(row['mpi_last3_sec'])} | {row['speedup_last3']:.4f}x |"
    )
lines += [
    "",
    "Raw per-run logs are in `logs/`; per-run numeric data is in `per_run.csv`.",
]
(out / "summary.md").write_text("\n".join(lines) + "\n")
PY

echo "finished_runs_at=$(date -Is)" >> "$OUT/meta.txt"
find "$OUT" -maxdepth 1 -name '*.json' | wc -l > "$OUT/json_count.txt"
echo "DONE $(date -Is)" >> "$OUT/progress.log"
