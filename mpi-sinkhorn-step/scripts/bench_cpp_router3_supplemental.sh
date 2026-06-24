#!/usr/bin/env bash
set -u

ROOT="${ROOT:-$HOME/Parallel-Programming-Project/mpi-sinkhorn-step}"
BIN="$ROOT/cpp/sinkhorn_cpp"
OUT="${OUT:-$ROOT/results/bench_cpp_router3_supplemental_after_ram}"
PREVIOUS_OUT="${PREVIOUS_OUT:-$ROOT/results/bench_cpp_router3_report_suite_safe}"
LOG="$OUT/logs"
HOSTFILE="${HOSTFILE:-$HOME/hosts_router_3_slots6}"
SSH_AGENT="${SSH_AGENT:-ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null}"
WORKER_HOSTS="${WORKER_HOSTS:-192.168.1.26 192.168.1.75}"
HOST_LINES="${HOST_LINES:-192.168.1.25 slots=6
192.168.1.26 slots=6
192.168.1.75 slots=6}"
BENCHMARK_NAME="${BENCHMARK_NAME:-bench_cpp_router3_supplemental_after_ram}"
HOSTS_LABEL="${HOSTS_LABEL:-.25,.26,.75}"
NETWORK_LABEL="${NETWORK_LABEL:-router_ethernet}"
NETWORK_IF="${NETWORK_IF:-enp0s3}"
HOST_SLOTS="${HOST_SLOTS:-6}"
HOSTFILE_SNAPSHOT_NAME="${HOSTFILE_SNAPSHOT_NAME:-hosts_router_3_slots6.txt}"
SUPPLEMENTAL_NOTE="${SUPPLEMENTAL_NOTE:-after RAM upgrade and 6GB swap per node}"
PARTIAL_NOTE="${PARTIAL_NOTE:-the retained np=12 run at N=8000 is partial because the third repetition made the master VM stop accepting SSH; the table keeps the completed JSON files and exposes count}"
if [ -z "${MPI_NET+x}" ]; then
  MPI_NET="--mca btl tcp,self --mca btl_tcp_if_include enp0s3 --mca oob_tcp_if_include enp0s3"
fi

PROCESS_EXTRA_NPS="${PROCESS_EXTRA_NPS:-8}"
PROCESS_EXTRA_REPS="${PROCESS_EXTRA_REPS:-3}"
SPEEDUP2N_NPS="${SPEEDUP2N_NPS:-1 2 4 8}"
SPEEDUP2N_REPS="${SPEEDUP2N_REPS:-1}"
SIZE_PROBE_SIZES="${SIZE_PROBE_SIZES:-14000 16000 18000}"
SIZE_PROBE_NP="${SIZE_PROBE_NP:-3}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"

mkdir -p "$LOG"
rm -f "$OUT/failures.txt"

make -C "$ROOT/cpp"
for host in $WORKER_HOSTS; do
  ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "tung@$host" "mkdir -p '$ROOT/cpp' '$ROOT/scripts'"
  rsync -az -e "$SSH_AGENT" "$BIN" "tung@$host:$BIN"
done

printf "%s\n" "$HOST_LINES" > "$HOSTFILE"
cp "$HOSTFILE" "$OUT/$HOSTFILE_SNAPSHOT_NAME"
{
  echo "benchmark=$BENCHMARK_NAME"
  echo "started_at=$(date -Is)"
  echo "root=$ROOT"
  echo "binary=$BIN"
  echo "hostfile=$HOSTFILE"
  echo "hosts=$HOSTS_LABEL"
  echo "network=$NETWORK_LABEL"
  echo "network_if=$NETWORK_IF"
  echo "host_slots=$HOST_SLOTS"
  echo "supplemental_note=$SUPPLEMENTAL_NOTE"
  echo "partial_note=$PARTIAL_NOTE"
  echo "process_extra_size=8000"
  echo "process_extra_nps=$PROCESS_EXTRA_NPS"
  echo "process_extra_reps=$PROCESS_EXTRA_REPS"
  echo "speedup2n_size=16000"
  echo "speedup2n_nps=$SPEEDUP2N_NPS"
  echo "speedup2n_reps=$SPEEDUP2N_REPS"
  echo "size_probe_sizes=$SIZE_PROBE_SIZES"
  echo "size_probe_np=$SIZE_PROBE_NP"
  echo "max_iters=50"
  echo "epsilon=0.1"
  echo "seed=0"
  echo "cost_mode=random"
} > "$OUT/meta.txt"

cat > "$OUT/run_manifest.md" <<EOF
# $BENCHMARK_NAME Supplemental Manifest

| group | config | purpose |
|---|---|---|
| process_extra | N=8000, np=$PROCESS_EXTRA_NPS, reps=$PROCESS_EXTRA_REPS, iters=50 | Add higher process counts missing from the first process sweep |
| speedup2n | N=16000, seq + np=$SPEEDUP2N_NPS, reps=$SPEEDUP2N_REPS, iters=50 | Speedup at 2N |
| workload_balance_extract | N=8000, np=3, iters=2000 | Extract per-rank load-balance table from previous suite |
| size_probe | N=$SIZE_PROBE_SIZES, np=$SIZE_PROBE_NP, reps=1, iters=50 | Check larger dense sizes after RAM upgrade |
EOF

if [ -s "$OUT/progress.log" ]; then
  echo "RESUME $(date -Is)" >> "$OUT/progress.log"
else
  echo "SMOKE $(date -Is)" > "$OUT/progress.log"
fi
mpirun -np 3 --hostfile "$HOSTFILE" --map-by node $MPI_NET --mca plm_rsh_agent "$SSH_AGENT" hostname </dev/null > "$OUT/mpi_hostname_smoke.log" 2>&1 || {
  echo "smoke_failed" >> "$OUT/failures.txt"
  cat "$OUT/mpi_hostname_smoke.log" >> "$OUT/progress.log"
  exit 1
}

run_seq() {
  local experiment="$1"
  local n="$2"
  local iters="$3"
  local rep="$4"
  local json="$OUT/${experiment}_seq_n${n}_iters${iters}_rep${rep}.json"
  local log="$LOG/${experiment}_seq_n${n}_iters${iters}_rep${rep}.log"
  local rc=0
  if [ -s "$json" ]; then
    echo "SKIP $experiment seq n=$n iters=$iters rep=$rep $(date -Is)" >> "$OUT/progress.log"
    return 0
  fi
  echo "RUN $experiment seq n=$n iters=$iters rep=$rep $(date -Is)" >> "$OUT/progress.log"
  {
    echo "experiment=$experiment"
    echo "kind=seq"
    echo "size=$n"
    echo "max_iters=$iters"
    echo "rep=$rep"
    echo "start=$(date -Is)"
    /usr/bin/time -f "wall_clock_sec=%e max_rss_kb=%M" timeout -k 20s "$TIMEOUT_SEC" \
      "$BIN" --mode sequential --rows "$n" --cols "$n" --cost-mode random --seed 0 \
      --epsilon 0.1 --max-iters "$iters" --tol 0 --check-every "$iters" --output "$json"
    rc=$?
    echo "exit_code=$rc"
    echo "end=$(date -Is)"
  } > "$log" 2>&1
  if [ "$rc" -ne 0 ]; then
    echo "FAIL $experiment seq n=$n iters=$iters rep=$rep rc=$rc $(date -Is)" | tee -a "$OUT/failures.txt" >> "$OUT/progress.log"
  fi
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
    /usr/bin/time -f "wall_clock_sec=%e max_rss_kb=%M" timeout -k 20s "$TIMEOUT_SEC" \
      mpirun -np "$np" --hostfile "$HOSTFILE" --map-by node $MPI_NET \
      --mca plm_rsh_agent "$SSH_AGENT" \
      "$BIN" --mode mpi --rows "$n" --cols "$n" --cost-mode random --seed 0 \
      --epsilon 0.1 --max-iters "$iters" --tol 0 --check-every "$iters" \
      --comm-mode double --output "$json" </dev/null
    rc=$?
    echo "exit_code=$rc"
    echo "end=$(date -Is)"
  } > "$log" 2>&1
  if [ "$rc" -ne 0 ]; then
    echo "FAIL $experiment mpi n=$n np=$np iters=$iters rep=$rep rc=$rc $(date -Is)" | tee -a "$OUT/failures.txt" >> "$OUT/progress.log"
  fi
}

for rep in $(seq 1 "$PROCESS_EXTRA_REPS"); do
  for np in $PROCESS_EXTRA_NPS; do
    run_mpi "process_extra" 8000 "$np" 50 "$rep"
  done
done

for rep in $(seq 1 "$SPEEDUP2N_REPS"); do
  run_seq "speedup2n" 16000 50 "$rep"
  for np in $SPEEDUP2N_NPS; do
    run_mpi "speedup2n" 16000 "$np" 50 "$rep"
  done
done

for n in $SIZE_PROBE_SIZES; do
  run_mpi "size_probe" "$n" "$SIZE_PROBE_NP" 50 1
done

python3 "$ROOT/scripts/summarize_cpp_router3_supplemental.py" "$OUT" "$PREVIOUS_OUT" || {
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
