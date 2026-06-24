#!/usr/bin/env bash
set -u

ROOT="${ROOT:-$HOME/Parallel-Programming-Project/mpi-sinkhorn-step}"
BIN="$ROOT/cpp/sinkhorn_cpp"
OUT="${OUT:-$ROOT/results/bench_cpp_router3_dynamic_fixed50_reps4}"
LOG="$OUT/logs"
HOSTFILE="${HOSTFILE:-$HOME/hosts_router_3}"
HOST_LINES="${HOST_LINES:-192.168.1.25 slots=2
192.168.1.26 slots=2
192.168.1.75 slots=2}"
WORKER_HOSTS="${WORKER_HOSTS:-192.168.1.26 192.168.1.75}"
SSH_AGENT="${SSH_AGENT:-ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null}"
if [ -z "${MPI_NET+x}" ]; then
  MPI_NET="--mca btl tcp,self --mca btl_tcp_if_include enp0s3 --mca oob_tcp_if_include enp0s3"
fi
NP="${NP:-3}"
SIZES="${SIZES:-2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000}"
REPS="${REPS:-4}"
RUNTIME_SIZES="${RUNTIME_SIZES:-2000 4000}"
RUNTIME_REPS="${RUNTIME_REPS:-2}"
CHUNK_ROWS="${CHUNK_ROWS:-256}"
ADAPT_ALPHA="${ADAPT_ALPHA:-0.5}"
MANUAL_WEIGHTS="${MANUAL_WEIGHTS:-2,1,1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"

mkdir -p "$LOG"
rm -f "$OUT/failures.txt"
make -C "$ROOT/cpp"
for host in $WORKER_HOSTS; do
  ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "tung@$host" "mkdir -p '$ROOT/cpp'"
  rsync -az -e "$SSH_AGENT" "$BIN" "tung@$host:$BIN"
done

printf "%s\n" "$HOST_LINES" > "$HOSTFILE"
cp "$HOSTFILE" "$OUT/hosts_router_3.txt"

equal_weights() {
  python3 - "$NP" <<'PY'
import sys
np = int(sys.argv[1])
print(",".join(["1"] * np))
PY
}

json_suggested_weights() {
  python3 - "$1" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
    print(data.get("suggested_rank_weights_csv", ""))
except Exception:
    print("")
PY
}

smooth_weights() {
  python3 - "$1" "$2" "$ADAPT_ALPHA" <<'PY'
import json, sys
path, current_csv, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
current = [float(x) for x in current_csv.split(",") if x]
try:
    data = json.load(open(path))
    suggested = [float(x) for x in data.get("suggested_rank_weights", [])]
except Exception:
    suggested = []
if len(suggested) != len(current) or not suggested:
    print(current_csv)
    raise SystemExit
updated = [(1.0 - alpha) * c + alpha * s for c, s in zip(current, suggested)]
mean = sum(updated) / len(updated)
updated = [max(0.05, x / mean) for x in updated]
mean = sum(updated) / len(updated)
print(",".join(f"{x / mean:.10g}" for x in updated))
PY
}

{
  echo "benchmark=bench_cpp_router3_dynamic_fixed50_reps4"
  echo "started_at=$(date -Is)"
  echo "root=$ROOT"
  echo "binary=$BIN"
  echo "hostfile=$HOSTFILE"
  echo "hosts=.25,.26,.75"
  echo "sizes=$SIZES"
  echo "reps=$REPS"
  echo "runtime_queue_sizes=$RUNTIME_SIZES"
  echo "runtime_queue_reps=$RUNTIME_REPS"
  echo "mpi_np=$NP"
  echo "chunk_rows=$CHUNK_ROWS"
  echo "adapt_alpha=$ADAPT_ALPHA"
  echo "manual_weights=$MANUAL_WEIGHTS"
  echo "max_iters=50"
  echo "epsilon=0.1"
  echo "seed=0"
  echo "cost_mode=random"
} > "$OUT/meta.txt"

cat > "$OUT/run_manifest.md" <<EOF
# Dynamic Router3 Run Manifest

Cluster: \`.25/.26/.75\`, MPI \`np=$NP\`, fixed 50 Sinkhorn iterations, chunk rows \`$CHUNK_ROWS\`.

| variant | partition mode | sizes | reps | weights | purpose |
|---|---|---|---:|---|---|
| seq | sequential | $SIZES | $REPS | n/a | Fresh sequential baseline on master |
| contiguous | contiguous | $SIZES | $REPS | n/a | Backward-compatible MPI baseline |
| weighted_equal | weighted-chunk | $SIZES | $REPS | equal | Chunk scheduler without hetero weighting |
| weighted_manual_211 | weighted-chunk | $SIZES | $REPS | $MANUAL_WEIGHTS | Explicit CLI rank weights coverage |
| weighted_adaptive | weighted-chunk | $SIZES | $REPS | learned across reps | Adaptive weight feedback coverage |
| runtime_queue | runtime-queue | $RUNTIME_SIZES | $RUNTIME_REPS | equal | Experimental queue overhead/correctness smoke |
EOF

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
    echo "variant=seq"
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

run_mpi_variant() {
  local variant="$1"
  local partition="$2"
  local n="$3"
  local rep="$4"
  local weights="${5:-}"
  local json="$OUT/${variant}_${n}_np${NP}_rep${rep}.json"
  local log="$LOG/${variant}_${n}_np${NP}_rep${rep}.log"
  local rc=0
  [ -s "$json" ] && { echo "SKIP $variant n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"; return 0; }
  echo "RUN $variant n=$n rep=$rep weights=${weights:-none} $(date -Is)" >> "$OUT/progress.log"

  local weight_args=()
  if [ -n "$weights" ]; then
    weight_args=(--rank-weights "$weights")
  fi

  {
    echo "variant=$variant"
    echo "partition=$partition"
    echo "size=$n"
    echo "rep=$rep"
    echo "np=$NP"
    echo "weights=${weights:-}"
    echo "start=$(date -Is)"
    echo "cmd=mpirun -np $NP --hostfile $HOSTFILE --map-by node $MPI_NET --mca plm_rsh_agent $SSH_AGENT $BIN --mode mpi --rows $n --cols $n --cost-mode random --seed 0 --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --comm-mode double --partition-mode $partition --chunk-rows $CHUNK_ROWS ${weight_args[*]} --output $json"
    /usr/bin/time -f "wall_clock_sec=%e max_rss_kb=%M" timeout -k 20s "$TIMEOUT_SEC"s \
      mpirun -np "$NP" --hostfile "$HOSTFILE" --map-by node $MPI_NET \
      --mca plm_rsh_agent "$SSH_AGENT" \
      "$BIN" --mode mpi --rows "$n" --cols "$n" --cost-mode random --seed 0 \
      --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --comm-mode double \
      --partition-mode "$partition" --chunk-rows "$CHUNK_ROWS" "${weight_args[@]}" --output "$json" </dev/null
    rc=$?
    echo "exit_code=$rc"
    echo "end=$(date -Is)"
  } > "$log" 2>&1
  if [ "$rc" -ne 0 ]; then
    echo "FAIL $variant n=$n rep=$rep rc=$rc $(date -Is)" | tee -a "$OUT/failures.txt" >> "$OUT/progress.log"
  fi
}

EQUAL_WEIGHTS="$(equal_weights)"
INITIAL_WEIGHTS="${INITIAL_WEIGHTS:-$EQUAL_WEIGHTS}"
declare -A ADAPTIVE_WEIGHTS
for n in $SIZES; do
  ADAPTIVE_WEIGHTS[$n]="$INITIAL_WEIGHTS"
done

echo "size,rep,weights_used_csv,suggested_csv,next_weights_csv" > "$OUT/adaptive_weights_by_rep.csv"

for rep in $(seq 1 "$REPS"); do
  for n in $SIZES; do
    run_seq "$n" "$rep"
  done
  for n in $SIZES; do
    run_mpi_variant "contiguous" "contiguous" "$n" "$rep" ""
  done
  for n in $SIZES; do
    run_mpi_variant "weighted_equal" "weighted-chunk" "$n" "$rep" "$EQUAL_WEIGHTS"
  done
  if [ "$(python3 - "$MANUAL_WEIGHTS" <<'PY'
import sys
print(len([x for x in sys.argv[1].split(',') if x.strip()]))
PY
)" -eq "$NP" ]; then
    for n in $SIZES; do
      run_mpi_variant "weighted_manual_211" "weighted-chunk" "$n" "$rep" "$MANUAL_WEIGHTS"
    done
  else
    echo "SKIP weighted_manual_211 rep=$rep because manual weight count does not match NP=$NP" >> "$OUT/progress.log"
  fi
  for n in $SIZES; do
    weights="${ADAPTIVE_WEIGHTS[$n]}"
    run_mpi_variant "weighted_adaptive" "weighted-chunk" "$n" "$rep" "$weights"
    json="$OUT/weighted_adaptive_${n}_np${NP}_rep${rep}.json"
    suggested="$(json_suggested_weights "$json")"
    next="$(smooth_weights "$json" "$weights")"
    echo "$n,$rep,$weights,$suggested,$next" >> "$OUT/adaptive_weights_by_rep.csv"
    ADAPTIVE_WEIGHTS[$n]="$next"
  done
done

for rep in $(seq 1 "$RUNTIME_REPS"); do
  for n in $RUNTIME_SIZES; do
    run_mpi_variant "runtime_queue" "runtime-queue" "$n" "$rep" ""
  done
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
pattern = re.compile(r"^(seq|contiguous|weighted_equal|weighted_manual_211|weighted_adaptive|runtime_queue)_(\d+)(?:_np\d+)?_rep(\d+)\.json$")
rows = []

for path in sorted(out.glob("*.json")):
    match = pattern.match(path.name)
    if not match:
        continue
    variant, size, rep = match.group(1), int(match.group(2)), int(match.group(3))
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    rows.append({
        "variant": variant,
        "size": size,
        "rep": rep,
        "runtime_sec": float(data["runtime_sec"]),
        "iterations": int(data.get("iterations", 0)),
        "row_error": float(data.get("row_error", math.nan)),
        "col_error": float(data.get("col_error", math.nan)),
        "objective": float(data.get("transport_objective", math.nan)),
        "num_processes": int(data.get("num_processes", 1)),
        "partition_mode": data.get("partition_mode", ""),
        "rank_weights_csv": ",".join(str(x) for x in data.get("rank_weights", [])),
        "suggested_rank_weights_csv": data.get("suggested_rank_weights_csv", ""),
        "json": str(path),
        "log": str(out / "logs" / path.name.replace(".json", ".log")),
    })

with (out / "per_run.csv").open("w", newline="") as fh:
    fields = [
        "variant", "size", "rep", "runtime_sec", "iterations", "row_error", "col_error",
        "objective", "num_processes", "partition_mode", "rank_weights_csv",
        "suggested_rank_weights_csv", "json", "log",
    ]
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

seq_by_size = {}
seq_obj_by_size_rep = {}
for row in rows:
    if row["variant"] == "seq":
        seq_by_size.setdefault(row["size"], []).append(row["runtime_sec"])
        seq_obj_by_size_rep[(row["size"], row["rep"])] = row["objective"]

summary = []
for variant in sorted({row["variant"] for row in rows}):
    for size in sorted({row["size"] for row in rows if row["variant"] == variant}):
        group = sorted([row for row in rows if row["variant"] == variant and row["size"] == size], key=lambda x: x["rep"])
        runtimes = [row["runtime_sec"] for row in group]
        seq_avg = statistics.mean(seq_by_size.get(size, [])) if seq_by_size.get(size) else math.nan
        avg_runtime = statistics.mean(runtimes)
        first3 = statistics.mean(runtimes[:3]) if len(runtimes) >= 3 else math.nan
        last3 = statistics.mean(runtimes[-3:]) if len(runtimes) >= 3 else math.nan
        speedup = seq_avg / avg_runtime if variant != "seq" and seq_avg and not math.isnan(seq_avg) else math.nan
        obj_diffs = [
            abs(row["objective"] - seq_obj_by_size_rep[(size, row["rep"])])
            for row in group
            if (size, row["rep"]) in seq_obj_by_size_rep and not math.isnan(row["objective"])
        ]
        summary.append({
            "variant": variant,
            "size": size,
            "count": len(group),
            "avg_runtime_sec": avg_runtime,
            "first3_runtime_sec": first3,
            "last3_runtime_sec": last3,
            "seq_avg_runtime_sec": seq_avg,
            "speedup_avg": speedup,
            "efficiency": speedup / np if variant != "seq" and not math.isnan(speedup) else math.nan,
            "avg_objective_diff": statistics.mean(obj_diffs) if obj_diffs else math.nan,
            "last_suggested_weights_csv": group[-1]["suggested_rank_weights_csv"],
        })

with (out / "summary_by_variant.csv").open("w", newline="") as fh:
    fields = [
        "variant", "size", "count", "avg_runtime_sec", "first3_runtime_sec", "last3_runtime_sec",
        "seq_avg_runtime_sec", "speedup_avg", "efficiency", "avg_objective_diff",
        "last_suggested_weights_csv",
    ]
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for row in summary:
        writer.writerow(row)

def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

lines = [
    "# C++ Sinkhorn Router3 Dynamic Scheduling Summary",
    "",
    f"NP: `{np}`, fixed `50` iterations, random cost seed `0`, epsilon `0.1`.",
    "",
    "| variant | size | count | avg runtime s | first3 s | last3 s | speedup | efficiency | avg obj diff | last suggested weights |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in summary:
    lines.append(
        "| {variant} | {size} | {count} | {avg} | {first3} | {last3} | {speedup} | {eff} | {diff} | `{weights}` |".format(
            variant=row["variant"],
            size=row["size"],
            count=row["count"],
            avg=fmt(row["avg_runtime_sec"]),
            first3=fmt(row["first3_runtime_sec"]),
            last3=fmt(row["last3_runtime_sec"]),
            speedup=fmt(row["speedup_avg"]),
            eff=fmt(row["efficiency"]),
            diff=fmt(row["avg_objective_diff"]),
            weights=row["last_suggested_weights_csv"],
        )
    )
lines += [
    "",
    "Adaptive weights used per rep are in `adaptive_weights_by_rep.csv`.",
    "Raw per-run data is in `per_run.csv`; logs are in `logs/`.",
]
(out / "summary.md").write_text("\n".join(lines) + "\n")
PY

echo "finished_at=$(date -Is)" >> "$OUT/meta.txt"
echo "DONE $(date -Is)" >> "$OUT/progress.log"
