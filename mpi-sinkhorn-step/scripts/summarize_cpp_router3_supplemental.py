#!/usr/bin/env python3
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


PATTERN = re.compile(
    r"^(?P<experiment>[a-z0-9_]+)_(?P<kind>seq|mpi)_n(?P<size>\d+)"
    r"(?:_np(?P<np>\d+))?_iters(?P<iters>\d+)_rep(?P<rep>\d+)\.json$"
)


def mean(values):
    return statistics.mean(values) if values else math.nan


def fmt(value, digits=6):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def load_rows(out):
    rows = []
    for path in sorted(out.glob("*.json")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        rank_compute = data.get("rank_compute_sec") or []
        rank_comm = data.get("rank_comm_sec") or []
        rank_total = data.get("rank_total_sec") or []
        totals = rank_total or [float(data["runtime_sec"])]
        rows.append(
            {
                "experiment": match.group("experiment"),
                "kind": match.group("kind"),
                "size": int(match.group("size")),
                "np": int(match.group("np") or data.get("num_processes", 1)),
                "max_iters": int(match.group("iters")),
                "rep": int(match.group("rep")),
                "runtime_sec": float(data["runtime_sec"]),
                "objective": float(data["transport_objective"]),
                "row_error": float(data["row_error"]),
                "col_error": float(data["col_error"]),
                "column_syncs": int(data.get("column_syncs", 0)),
                "column_payload_bytes": int(data.get("column_payload_bytes", 0)),
                "hostnames": ";".join(data.get("hostnames", [])),
                "local_rows_by_rank": json.dumps(data.get("local_rows_by_rank") or []),
                "rank_compute_sec": json.dumps(rank_compute),
                "rank_comm_sec": json.dumps(rank_comm),
                "rank_total_sec": json.dumps(rank_total),
                "max_rank_compute_sec": max(rank_compute) if rank_compute else math.nan,
                "max_rank_comm_sec": max(rank_comm) if rank_comm else math.nan,
                "max_rank_total_sec": max(totals),
                "min_rank_total_sec": min(totals),
                "rank_imbalance_pct": (
                    100.0 * (max(totals) - min(totals)) / max(totals)
                    if totals and max(totals) > 0
                    else 0.0
                ),
                "json": str(path),
            }
        )
    return rows


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    return groups


def read_meta(out):
    meta = {}
    path = out / "meta.txt"
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def previous_seq_avg(previous, size):
    paths = sorted(previous.glob(f"baseline_seq_n{size}_iters50_rep*.json"))
    values = [float(json.loads(path.read_text())["runtime_sec"]) for path in paths]
    return mean(values)


def previous_workload_rows(previous):
    path = previous / "workload_calibration_mpi_n8000_np3_iters2000_rep1.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows = []
    totals = data.get("rank_total_sec") or []
    computes = data.get("rank_compute_sec") or []
    comms = data.get("rank_comm_sec") or []
    local_rows = data.get("local_rows_by_rank") or []
    for rank, total in enumerate(totals):
        compute = computes[rank] if rank < len(computes) else math.nan
        comm = comms[rank] if rank < len(comms) else math.nan
        rows.append(
            {
                "experiment": "workload_balance",
                "size": 8000,
                "max_iters": 2000,
                "np": 3,
                "rank": rank,
                "local_rows": local_rows[rank] if rank < len(local_rows) else "",
                "total_sec": total,
                "compute_sec": compute,
                "comm_sec": comm,
                "overhead_sec": max(0.0, total - compute - comm)
                if not math.isnan(compute) and not math.isnan(comm)
                else math.nan,
            }
        )
    return rows


def summarize_process_extra(rows, previous):
    seq8000 = previous_seq_avg(previous, 8000)
    summary = []
    for (np, size, iters), group in sorted(
        grouped([r for r in rows if r["experiment"] == "process_extra"], ["np", "size", "max_iters"]).items()
    ):
        runtime = mean([r["runtime_sec"] for r in group])
        summary.append(
            {
                "size": size,
                "max_iters": iters,
                "np": np,
                "count": len(group),
                "seq8000_avg_sec": seq8000,
                "mpi_avg_sec": runtime,
                "speedup_vs_seq8000": seq8000 / runtime if runtime and seq8000 else math.nan,
                "efficiency": (seq8000 / runtime) / np if runtime and seq8000 else math.nan,
                "avg_rank_imbalance_pct": mean([r["rank_imbalance_pct"] for r in group]),
                "avg_max_rank_compute_sec": mean([r["max_rank_compute_sec"] for r in group]),
                "avg_max_rank_comm_sec": mean([r["max_rank_comm_sec"] for r in group]),
            }
        )
    return summary


def summarize_speedup_2n(rows):
    seq = [r for r in rows if r["experiment"] == "speedup2n" and r["kind"] == "seq"]
    seq_avg = mean([r["runtime_sec"] for r in seq])
    summary = []
    for (np, size, iters), group in sorted(
        grouped([r for r in rows if r["experiment"] == "speedup2n" and r["kind"] == "mpi"], ["np", "size", "max_iters"]).items()
    ):
        runtime = mean([r["runtime_sec"] for r in group])
        summary.append(
            {
                "size": size,
                "max_iters": iters,
                "np": np,
                "count": len(group),
                "seq_avg_sec": seq_avg,
                "mpi_avg_sec": runtime,
                "speedup_vs_seq": seq_avg / runtime if seq_avg and runtime else math.nan,
                "efficiency": (seq_avg / runtime) / np if seq_avg and runtime else math.nan,
                "avg_rank_imbalance_pct": mean([r["rank_imbalance_pct"] for r in group]),
                "avg_max_rank_compute_sec": mean([r["max_rank_compute_sec"] for r in group]),
                "avg_max_rank_comm_sec": mean([r["max_rank_comm_sec"] for r in group]),
            }
        )
    return summary


def summarize_size_probe(rows):
    summary = []
    for (size, np, iters), group in sorted(
        grouped([r for r in rows if r["experiment"] == "size_probe"], ["size", "np", "max_iters"]).items()
    ):
        runtime = mean([r["runtime_sec"] for r in group])
        summary.append(
            {
                "size": size,
                "max_iters": iters,
                "np": np,
                "count": len(group),
                "mpi_avg_sec": runtime,
                "avg_rank_imbalance_pct": mean([r["rank_imbalance_pct"] for r in group]),
                "avg_max_rank_compute_sec": mean([r["max_rank_compute_sec"] for r in group]),
                "avg_max_rank_comm_sec": mean([r["max_rank_comm_sec"] for r in group]),
            }
        )
    return summary


def markdown(out, process_extra, speedup2n, size_probe, workload_balance):
    meta = read_meta(out)
    benchmark = meta.get("benchmark", "bench_cpp_router3_supplemental_after_ram")
    hosts = meta.get("hosts", ".25,.26,.75")
    network = meta.get("network", "router_ethernet")
    host_slots = meta.get("host_slots", "6")
    note = meta.get("supplemental_note", "after RAM upgrade and 6GB swap per node")
    partial_note = meta.get(
        "partial_note",
        "the retained np=12 run at N=8000 is partial because the third repetition made the master VM stop accepting SSH; the table keeps the completed JSON files and exposes count",
    )
    process_extra_nps = meta.get("process_extra_nps", "8")
    speedup2n_nps = meta.get("speedup2n_nps", "1 2 4 8")
    size_probe_sizes = meta.get("size_probe_sizes", "14000 16000 18000")
    size_probe_np = meta.get("size_probe_np", "3")
    lines = [
        f"# C++ Sinkhorn {benchmark} Supplemental Experiments",
        "",
        f"Cluster: `{hosts}` over `{network}` with `{host_slots}` slots per node; {note}.",
        "",
        "This supplement covers the experiments still missing after the first report suite:",
        f"- Extra process counts at `N=8000`: `np={process_extra_nps}`.",
        f"- `2N` speedup at `N=16000`: sequential plus `np={speedup2n_nps}`.",
        "- Load-balance extraction at the report workload `N=8000`, `iters=2000`, `np=3`.",
        f"- Larger-size probe at `N={size_probe_sizes}`, `np={size_probe_np}`.",
        "",
        f"Note: {partial_note}.",
        "",
        "## Extra Process Counts At N=8000",
        "",
        "| np | count | mpi avg s | speedup vs seq8000 | efficiency | avg imbalance % | max compute s | max comm s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in process_extra:
        lines.append(
            f"| {row['np']} | {row['count']} | {fmt(row['mpi_avg_sec'])} | {fmt(row['speedup_vs_seq8000'])}x | "
            f"{fmt(row['efficiency'])} | {fmt(row['avg_rank_imbalance_pct'])} | "
            f"{fmt(row['avg_max_rank_compute_sec'])} | {fmt(row['avg_max_rank_comm_sec'])} |"
        )
    lines += [
        "",
        "## 2N Speedup At N=16000",
        "",
        "| np | count | mpi avg s | speedup vs seq | efficiency | avg imbalance % | max compute s | max comm s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in speedup2n:
        lines.append(
            f"| {row['np']} | {row['count']} | {fmt(row['mpi_avg_sec'])} | {fmt(row['speedup_vs_seq'])}x | "
            f"{fmt(row['efficiency'])} | {fmt(row['avg_rank_imbalance_pct'])} | "
            f"{fmt(row['avg_max_rank_compute_sec'])} | {fmt(row['avg_max_rank_comm_sec'])} |"
        )
    lines += [
        "",
        "## Larger Size Probe",
        "",
        "| size | np | runtime s | avg imbalance % | max compute s | max comm s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in size_probe:
        lines.append(
            f"| {row['size']} | {row['np']} | {fmt(row['mpi_avg_sec'])} | "
            f"{fmt(row['avg_rank_imbalance_pct'])} | {fmt(row['avg_max_rank_compute_sec'])} | "
            f"{fmt(row['avg_max_rank_comm_sec'])} |"
        )
    lines += [
        "",
        "## Workload Load-Balance Extraction",
        "",
        "| rank | local rows | total s | compute s | comm s | overhead s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in workload_balance:
        lines.append(
            f"| {row['rank']} | {row['local_rows']} | {fmt(row['total_sec'])} | "
            f"{fmt(row['compute_sec'])} | {fmt(row['comm_sec'])} | {fmt(row['overhead_sec'])} |"
        )
    lines += [
        "",
        "Raw per-run data is in `per_run.csv`.",
        "The extracted workload balance rows are in `workload_balance_by_rank.csv`.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    out = Path(sys.argv[1])
    previous = Path(sys.argv[2])
    rows = load_rows(out)
    write_csv(out / "per_run.csv", rows)
    process_extra = summarize_process_extra(rows, previous)
    speedup2n = summarize_speedup_2n(rows)
    size_probe = summarize_size_probe(rows)
    workload_balance = previous_workload_rows(previous)
    write_csv(out / "process_extra_summary.csv", process_extra)
    write_csv(out / "speedup2n_summary.csv", speedup2n)
    write_csv(out / "size_probe_summary.csv", size_probe)
    write_csv(out / "workload_balance_by_rank.csv", workload_balance)
    markdown(out, process_extra, speedup2n, size_probe, workload_balance)


if __name__ == "__main__":
    main()
