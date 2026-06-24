#!/usr/bin/env bash
set -u

ROOT="${ROOT:-$HOME/Parallel-Programming-Project/mpi-sinkhorn-step}"
BIN="$ROOT/cpp/sinkhorn_cpp"
OUT="${OUT:-$ROOT/results/bench_cpp_router3_report_suite_safe}"
LOG="$OUT/logs"
HOSTFILE="${HOSTFILE:-$HOME/hosts_router_3_slots2}"
SEQ_SOURCE="${SEQ_SOURCE:-$ROOT/results/bench_cpp_router3_25_26_75_fixed50_reps4}"
SUMMARIZER="$ROOT/scripts/summarize_cpp_router_report_suite.py"
SSH_AGENT="${SSH_AGENT:-ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null}"
WORKER_HOSTS="${WORKER_HOSTS:-192.168.1.26 192.168.1.75}"
HOST_LINES="${HOST_LINES:-192.168.1.25 slots=2
192.168.1.26 slots=2
192.168.1.75 slots=2}"

if [ -z "${MPI_NET+x}" ]; then
  MPI_NET="--mca btl tcp,self --mca btl_tcp_if_include enp0s3 --mca oob_tcp_if_include enp0s3"
fi

BASELINE_SIZES="${BASELINE_SIZES:-2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000}"
BASELINE_REPS="${BASELINE_REPS:-4}"
BASELINE_NP="${BASELINE_NP:-3}"
PROCESS_SIZE="${PROCESS_SIZE:-8000}"
PROCESS_NPS="${PROCESS_NPS:-1 2 3 4 6}"
PROCESS_REPS="${PROCESS_REPS:-3}"
INPUT_CALIBRATION_SIZES="${INPUT_CALIBRATION_SIZES:-8000 10000 12000}"
WORKLOAD_CALIBRATION_ITERS="${WORKLOAD_CALIBRATION_ITERS:-200 500 1000 2000}"
LOAD_BALANCE_NP="${LOAD_BALANCE_NP:-6}"
LOAD_BALANCE_REPS="${LOAD_BALANCE_REPS:-3}"
MAX_ITERS="${MAX_ITERS:-50}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
PARTITION_MODE="${PARTITION_MODE:-contiguous}"
CHUNK_ROWS="${CHUNK_ROWS:-256}"
RANK_WEIGHTS="${RANK_WEIGHTS:-}"

mkdir -p "$LOG" "$ROOT/scripts"
rm -f "$OUT/failures.txt"

make -C "$ROOT/cpp"
for host in $WORKER_HOSTS; do
  ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "tung@$host" "mkdir -p '$ROOT/cpp' '$ROOT/scripts'"
  rsync -az -e "$SSH_AGENT" "$BIN" "tung@$host:$BIN"
done

printf "%s\n" "$HOST_LINES" > "$HOSTFILE"
cp "$HOSTFILE" "$OUT/hosts_router_3_slots2.txt"
{
  echo "benchmark=bench_cpp_router3_report_suite_safe"
  echo "started_at=$(date -Is)"
  echo "root=$ROOT"
  echo "binary=$BIN"
  echo "hostfile=$HOSTFILE"
  echo "hosts=.25,.26,.75"
  echo "network=router_ethernet"
  echo "network_if=enp0s3"
  echo "host_slots=2"
  echo "safety_note=size calibration is capped at 12000 after node .75 soft-lockup during larger/more aggressive run"
  echo "baseline_sizes=$BASELINE_SIZES"
  echo "baseline_reps=$BASELINE_REPS"
  echo "baseline_np=$BASELINE_NP"
  echo "process_size=$PROCESS_SIZE"
  echo "process_nps=$PROCESS_NPS"
  echo "process_reps=$PROCESS_REPS"
  echo "input_calibration_sizes=$INPUT_CALIBRATION_SIZES"
  echo "workload_calibration_iters=$WORKLOAD_CALIBRATION_ITERS"
  echo "load_balance_np=$LOAD_BALANCE_NP"
  echo "load_balance_reps=$LOAD_BALANCE_REPS"
  echo "max_iters=$MAX_ITERS"
  echo "epsilon=0.1"
  echo "seed=0"
  echo "cost_mode=random"
  echo "partition_mode=$PARTITION_MODE"
  echo "chunk_rows=$CHUNK_ROWS"
  echo "rank_weights=$RANK_WEIGHTS"
} > "$OUT/meta.txt"

{
  echo "# Router3 Report Suite Manifest"
  echo
  echo "| experiment | config | purpose |"
  echo "|---|---|---|"
  echo "| baseline | np=$BASELINE_NP, sizes=$BASELINE_SIZES, reps=$BASELINE_REPS, iters=$MAX_ITERS | Runtime vs input size on router |"
  echo "| process_sweep | size=$PROCESS_SIZE, np=$PROCESS_NPS, reps=$PROCESS_REPS, iters=$MAX_ITERS | Speedup/efficiency vs process count |"
  echo "| input_calibration | np=$BASELINE_NP, sizes=$INPUT_CALIBRATION_SIZES, reps=1, iters=$MAX_ITERS | Larger-size calibration within RAM limits |"
  echo "| workload_calibration | np=$BASELINE_NP, size=$PROCESS_SIZE, iters=$WORKLOAD_CALIBRATION_ITERS, reps=1 | 2-3 minute workload calibration without unsafe dense RAM growth |"
  echo "| load_balance | np=$LOAD_BALANCE_NP, size=$PROCESS_SIZE, reps=$LOAD_BALANCE_REPS, iters=$MAX_ITERS | Per-rank granularity/load-balance snapshot |"
  echo
  echo "Dynamic partition config: \`PARTITION_MODE=$PARTITION_MODE\`, \`CHUNK_ROWS=$CHUNK_ROWS\`, \`RANK_WEIGHTS=$RANK_WEIGHTS\`."
} > "$OUT/run_manifest.md"

echo "SMOKE $(date -Is)" > "$OUT/progress.log"
mpirun -np "$BASELINE_NP" --hostfile "$HOSTFILE" --map-by node $MPI_NET --mca plm_rsh_agent "$SSH_AGENT" hostname </dev/null > "$OUT/mpi_hostname_smoke_np${BASELINE_NP}.log" 2>&1 || {
  echo "smoke_np${BASELINE_NP}_failed" >> "$OUT/failures.txt"
  cat "$OUT/mpi_hostname_smoke_np${BASELINE_NP}.log" >> "$OUT/progress.log"
  exit 1
}
mpirun -np "$LOAD_BALANCE_NP" --hostfile "$HOSTFILE" --map-by node $MPI_NET --mca plm_rsh_agent "$SSH_AGENT" hostname </dev/null > "$OUT/mpi_hostname_smoke_np${LOAD_BALANCE_NP}.log" 2>&1 || {
  echo "smoke_np${LOAD_BALANCE_NP}_failed" >> "$OUT/failures.txt"
  cat "$OUT/mpi_hostname_smoke_np${LOAD_BALANCE_NP}.log" >> "$OUT/progress.log"
  exit 1
}

copy_seq_baseline() {
  local n="$1"
  local rep="$2"
  local src="$SEQ_SOURCE/seq_${n}_rep${rep}.json"
  local dst="$OUT/baseline_seq_n${n}_iters${MAX_ITERS}_rep${rep}.json"
  if [ -s "$dst" ]; then
    echo "SKIP baseline seq copy n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"
    return 0
  fi
  if [ ! -s "$src" ]; then
    echo "MISSING baseline seq source n=$n rep=$rep src=$src $(date -Is)" | tee -a "$OUT/failures.txt" >> "$OUT/progress.log"
    return 1
  fi
  cp "$src" "$dst"
  echo "COPY baseline seq n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"
}

run_mpi() {
  local experiment="$1"
  local n="$2"
  local np="$3"
  local iters="$4"
  local rep="$5"
  local json="$OUT/${experiment}_mpi_n${n}_np${np}_iters${iters}_rep${rep}.json"
  local log="$LOG/${experiment}_mpi_n${n}_np${np}_iters${iters}_rep${rep}.log"
  local rc=0
  local weight_args=()
  if [ -n "$RANK_WEIGHTS" ]; then
    local weight_count
    weight_count="$(python3 - "$RANK_WEIGHTS" <<'PY'
import sys
print(len([x for x in sys.argv[1].split(',') if x.strip()]))
PY
)"
    if [ "$weight_count" -eq "$np" ]; then
      weight_args=(--rank-weights "$RANK_WEIGHTS")
    else
      echo "NOTE skip rank weights for $experiment n=$n np=$np because weight_count=$weight_count $(date -Is)" >> "$OUT/progress.log"
    fi
  fi
  if [ -s "$json" ]; then
    echo "SKIP $experiment mpi n=$n np=$np iters=$iters rep=$rep $(date -Is)" >> "$OUT/progress.log"
    return 0
  fi
  echo "RUN $experiment mpi n=$n np=$np iters=$iters rep=$rep $(date -Is)" >> "$OUT/progress.log"
  {
    echo "experiment=$experiment"
    echo "kind=mpi"
    echo "size=$n"
    echo "np=$np"
    echo "max_iters=$iters"
    echo "rep=$rep"
    echo "start=$(date -Is)"
    echo "cmd=mpirun -np $np --hostfile $HOSTFILE --map-by node $MPI_NET --mca plm_rsh_agent $SSH_AGENT $BIN --mode mpi --rows $n --cols $n --cost-mode random --seed 0 --epsilon 0.1 --max-iters $iters --tol 0 --check-every $iters --comm-mode double --partition-mode $PARTITION_MODE --chunk-rows $CHUNK_ROWS ${weight_args[*]} --output $json"
    /usr/bin/time -f "wall_clock_sec=%e max_rss_kb=%M" timeout -k 20s "$TIMEOUT_SEC" \
      mpirun -np "$np" --hostfile "$HOSTFILE" --map-by node $MPI_NET \
      --mca plm_rsh_agent "$SSH_AGENT" \
      "$BIN" --mode mpi --rows "$n" --cols "$n" --cost-mode random --seed 0 \
      --epsilon 0.1 --max-iters "$iters" --tol 0 --check-every "$iters" \
      --comm-mode double --partition-mode "$PARTITION_MODE" --chunk-rows "$CHUNK_ROWS" \
      "${weight_args[@]}" --output "$json" </dev/null
    rc=$?
    echo "exit_code=$rc"
    echo "end=$(date -Is)"
  } > "$log" 2>&1
  if [ "$rc" -ne 0 ]; then
    echo "FAIL $experiment mpi n=$n np=$np iters=$iters rep=$rep rc=$rc $(date -Is)" | tee -a "$OUT/failures.txt" >> "$OUT/progress.log"
  fi
}

for rep in $(seq 1 "$BASELINE_REPS"); do
  for n in $BASELINE_SIZES; do
    copy_seq_baseline "$n" "$rep"
  done
  for n in $BASELINE_SIZES; do
    run_mpi "baseline" "$n" "$BASELINE_NP" "$MAX_ITERS" "$rep"
  done
done

for rep in $(seq 1 "$PROCESS_REPS"); do
  for np in $PROCESS_NPS; do
    run_mpi "process_sweep" "$PROCESS_SIZE" "$np" "$MAX_ITERS" "$rep"
  done
done

for n in $INPUT_CALIBRATION_SIZES; do
  run_mpi "input_calibration" "$n" "$BASELINE_NP" "$MAX_ITERS" "1"
done

for iters in $WORKLOAD_CALIBRATION_ITERS; do
  run_mpi "workload_calibration" "$PROCESS_SIZE" "$BASELINE_NP" "$iters" "1"
done

for rep in $(seq 1 "$LOAD_BALANCE_REPS"); do
  run_mpi "load_balance" "$PROCESS_SIZE" "$LOAD_BALANCE_NP" "$MAX_ITERS" "$rep"
done

python3 "$SUMMARIZER" "$OUT" || {
  echo "summary_failed" >> "$OUT/failures.txt"
  echo "SUMMARY_FAILED $(date -Is)" >> "$OUT/progress.log"
  exit 1
}

echo "finished_at=$(date -Is)" >> "$OUT/meta.txt"
find "$OUT" -maxdepth 1 -name '*.json' | wc -l > "$OUT/json_count.txt"
if [ -s "$OUT/failures.txt" ]; then
  echo "DONE_WITH_FAILURES $(date -Is)" >> "$OUT/progress.log"
  exit 1
fi
echo "DONE $(date -Is)" >> "$OUT/progress.log"
