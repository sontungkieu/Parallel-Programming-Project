#!/usr/bin/env bash
set -u
ROOT="$HOME/Parallel-Programming-Project/mpi-sinkhorn-step"
BIN="$ROOT/cpp/sinkhorn_cpp"
OUT="$ROOT/results/bench_cpp_router3_25_26_75_fixed50_reps4"
LOG="$OUT/logs"
HOSTFILE="$HOME/hosts_router_3"
SSH_AGENT="ssh -i /home/tung/.ssh/id_ed25519_mpi -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
MPI_NET="--mca btl tcp,self --mca btl_tcp_if_include enp0s3 --mca oob_tcp_if_include enp0s3"
mkdir -p "$LOG"
printf "%s\n" "192.168.1.25 slots=2" "192.168.1.26 slots=2" "192.168.1.75 slots=2" > "$HOSTFILE"
cp "$HOSTFILE" "$OUT/hosts_router_3.txt"
{
  echo "benchmark=bench_cpp_router3_25_26_75_fixed50_reps4"
  echo "started_at=$(date -Is)"
  echo "root=$ROOT"
  echo "binary=$BIN"
  echo "hostfile=$HOSTFILE"
  echo "hosts=.25,.26,.75"
  echo "sizes=2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000"
  echo "reps=4"
  echo "mpi_np=3"
  echo "map_by=node"
  echo "network_if=enp0s3"
  echo "max_iters=50"
  echo "epsilon=0.1"
  echo "seed=0"
  echo "cost_mode=random"
} > "$OUT/meta.txt"

echo "SMOKE $(date -Is)" > "$OUT/progress.log"
mpirun -np 3 --hostfile "$HOSTFILE" --map-by node $MPI_NET --mca plm_rsh_agent "$SSH_AGENT" hostname </dev/null > "$OUT/mpi_hostname_smoke.log" 2>&1 || {
  echo "smoke_failed" >> "$OUT/failures.txt"
  cat "$OUT/mpi_hostname_smoke.log" >> "$OUT/progress.log"
  exit 1
}

run_seq() {
  n="$1"; rep="$2"
  json="$OUT/seq_${n}_rep${rep}.json"
  log="$LOG/seq_${n}_rep${rep}.log"
  [ -s "$json" ] && { echo "SKIP seq n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"; return 0; }
  echo "RUN seq n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"
  {
    echo "kind=seq"
    echo "size=$n"
    echo "rep=$rep"
    echo "start=$(date -Is)"
    echo "cmd=$BIN --mode sequential --rows $n --cols $n --cost-mode random --seed 0 --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --output $json"
    /usr/bin/time -f "wall_clock_sec=%e max_rss_kb=%M" "$BIN" --mode sequential --rows "$n" --cols "$n" --cost-mode random --seed 0 --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --output "$json"
    rc=$?
    echo "exit_code=$rc"
    echo "end=$(date -Is)"
  } > "$log" 2>&1
  if [ "$rc" -ne 0 ]; then echo "FAIL seq n=$n rep=$rep rc=$rc $(date -Is)" | tee -a "$OUT/failures.txt" >> "$OUT/progress.log"; fi
}

run_mpi() {
  n="$1"; rep="$2"
  json="$OUT/mpi_${n}_np3_rep${rep}.json"
  log="$LOG/mpi_${n}_np3_rep${rep}.log"
  [ -s "$json" ] && { echo "SKIP mpi n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"; return 0; }
  echo "RUN mpi n=$n rep=$rep $(date -Is)" >> "$OUT/progress.log"
  {
    echo "kind=mpi"
    echo "size=$n"
    echo "rep=$rep"
    echo "np=3"
    echo "start=$(date -Is)"
    echo "cmd=mpirun -np 3 --hostfile $HOSTFILE --map-by node $MPI_NET --mca plm_rsh_agent $SSH_AGENT $BIN --mode mpi --rows $n --cols $n --cost-mode random --seed 0 --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --comm-mode double --output $json"
    /usr/bin/time -f "wall_clock_sec=%e max_rss_kb=%M" timeout -k 20s 300s mpirun -np 3 --hostfile "$HOSTFILE" --map-by node $MPI_NET --mca plm_rsh_agent "$SSH_AGENT" "$BIN" --mode mpi --rows "$n" --cols "$n" --cost-mode random --seed 0 --epsilon 0.1 --max-iters 50 --tol 0 --check-every 50 --comm-mode double --output "$json" </dev/null
    rc=$?
    echo "exit_code=$rc"
    echo "end=$(date -Is)"
  } > "$log" 2>&1
  if [ "$rc" -ne 0 ]; then echo "FAIL mpi n=$n rep=$rep rc=$rc $(date -Is)" | tee -a "$OUT/failures.txt" >> "$OUT/progress.log"; fi
}

SIZES="2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000"
for rep in 1 2 3 4; do
  for n in $SIZES; do run_seq "$n" "$rep"; done
  for n in $SIZES; do run_mpi "$n" "$rep"; done
done

echo "finished_runs_at=$(date -Is)" >> "$OUT/meta.txt"
find "$OUT" -maxdepth 1 -name '*.json' | wc -l > "$OUT/json_count.txt"
echo "DONE $(date -Is)" >> "$OUT/progress.log"
