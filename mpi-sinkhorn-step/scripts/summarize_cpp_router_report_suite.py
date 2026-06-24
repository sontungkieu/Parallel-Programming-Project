#!/usr/bin/env python3
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


PATTERN = re.compile(
    r"^(?P<experiment>[a-z_]+)_(?P<kind>seq|mpi)_n(?P<size>\d+)"
    r"(?:_np(?P<np>\d+))?_iters(?P<iters>\d+)_rep(?P<rep>\d+)\.json$"
)


def mean(values):
    return statistics.mean(values) if values else math.nan


def fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def as_list(data, key):
    value = data.get(key, [])
    return value if isinstance(value, list) else []


def read_rows(out):
    rows = []
    for path in sorted(out.glob("*.json")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        rank_compute = [float(x) for x in as_list(data, "rank_compute_sec")]
        rank_comm = [float(x) for x in as_list(data, "rank_comm_sec")]
        rank_total = [float(x) for x in as_list(data, "rank_total_sec")]
        local_rows = [int(x) for x in as_list(data, "local_rows_by_rank")]
        overhead = []
        for idx, total in enumerate(rank_total):
            compute = rank_compute[idx] if idx < len(rank_compute) else 0.0
            comm = rank_comm[idx] if idx < len(rank_comm) else 0.0
            overhead.append(max(0.0, total - compute - comm))
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
                "local_rows_by_rank": json.dumps(local_rows),
                "rank_compute_sec": json.dumps(rank_compute),
                "rank_comm_sec": json.dumps(rank_comm),
                "rank_total_sec": json.dumps(rank_total),
                "max_rank_compute_sec": max(rank_compute) if rank_compute else math.nan,
                "max_rank_comm_sec": max(rank_comm) if rank_comm else math.nan,
                "max_rank_overhead_sec": max(overhead) if overhead else math.nan,
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


def write_csv(path, rows, fields):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def union_fields(rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def grouped(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    return groups


def build_baseline(rows):
    out = []
    seq_groups = grouped(
        [r for r in rows if r["experiment"] == "baseline" and r["kind"] == "seq"],
        ["size", "max_iters"],
    )
    mpi_groups = grouped(
        [r for r in rows if r["experiment"] == "baseline" and r["kind"] == "mpi"],
        ["size", "max_iters", "np"],
    )
    for (size, iters, np), mpi in sorted(mpi_groups.items()):
        seq = seq_groups.get((size, iters), [])
        seq_by_rep = {r["rep"]: r for r in seq}
        seq_avg = mean([r["runtime_sec"] for r in seq])
        mpi_avg = mean([r["runtime_sec"] for r in mpi])
        obj_diffs = [
            abs(r["objective"] - seq_by_rep[r["rep"]]["objective"])
            for r in mpi
            if r["rep"] in seq_by_rep
        ]
        out.append(
            {
                "size": size,
                "max_iters": iters,
                "np": np,
                "seq_count": len(seq),
                "mpi_count": len(mpi),
                "seq_avg_sec": seq_avg,
                "mpi_avg_sec": mpi_avg,
                "speedup": seq_avg / mpi_avg if seq_avg and mpi_avg else math.nan,
                "efficiency": (seq_avg / mpi_avg) / np if seq_avg and mpi_avg else math.nan,
                "mpi_first3_sec": mean([r["runtime_sec"] for r in sorted(mpi, key=lambda x: x["rep"])[:3]]),
                "mpi_last3_sec": mean([r["runtime_sec"] for r in sorted(mpi, key=lambda x: x["rep"])[-3:]]),
                "max_obj_diff": max(obj_diffs) if obj_diffs else math.nan,
                "avg_rank_imbalance_pct": mean([r["rank_imbalance_pct"] for r in mpi]),
                "avg_max_rank_compute_sec": mean([r["max_rank_compute_sec"] for r in mpi]),
                "avg_max_rank_comm_sec": mean([r["max_rank_comm_sec"] for r in mpi]),
                "avg_max_rank_overhead_sec": mean([r["max_rank_overhead_sec"] for r in mpi]),
            }
        )
    return out


def build_process_sweep(rows):
    seq_groups = grouped(
        [
            r
            for r in rows
            if r["experiment"] == "baseline"
            and r["kind"] == "seq"
            and r["max_iters"] == 50
        ],
        ["size"],
    )
    out = []
    for (np, size, iters), group in sorted(
        grouped(
            [r for r in rows if r["experiment"] == "process_sweep" and r["kind"] == "mpi"],
            ["np", "size", "max_iters"],
        ).items()
    ):
        seq_avg = mean([r["runtime_sec"] for r in seq_groups.get((size,), [])])
        mpi_avg = mean([r["runtime_sec"] for r in group])
        out.append(
            {
                "size": size,
                "max_iters": iters,
                "np": np,
                "count": len(group),
                "seq_baseline_avg_sec": seq_avg,
                "mpi_avg_sec": mpi_avg,
                "speedup_vs_seq": seq_avg / mpi_avg if seq_avg and mpi_avg else math.nan,
                "efficiency_vs_seq": (seq_avg / mpi_avg) / np if seq_avg and mpi_avg else math.nan,
                "avg_rank_imbalance_pct": mean([r["rank_imbalance_pct"] for r in group]),
                "avg_max_rank_compute_sec": mean([r["max_rank_compute_sec"] for r in group]),
                "avg_max_rank_comm_sec": mean([r["max_rank_comm_sec"] for r in group]),
                "avg_max_rank_overhead_sec": mean([r["max_rank_overhead_sec"] for r in group]),
            }
        )
    return out


def build_simple_summary(rows, experiment):
    out = []
    for (size, iters, np), group in sorted(
        grouped(
            [r for r in rows if r["experiment"] == experiment and r["kind"] == "mpi"],
            ["size", "max_iters", "np"],
        ).items()
    ):
        out.append(
            {
                "experiment": experiment,
                "size": size,
                "max_iters": iters,
                "np": np,
                "count": len(group),
                "mpi_avg_sec": mean([r["runtime_sec"] for r in group]),
                "avg_rank_imbalance_pct": mean([r["rank_imbalance_pct"] for r in group]),
                "avg_max_rank_compute_sec": mean([r["max_rank_compute_sec"] for r in group]),
                "avg_max_rank_comm_sec": mean([r["max_rank_comm_sec"] for r in group]),
                "avg_max_rank_overhead_sec": mean([r["max_rank_overhead_sec"] for r in group]),
            }
        )
    return out


def build_load_balance_by_rank(rows):
    out = []
    for row in rows:
        if row["kind"] != "mpi" or row["experiment"] not in {"baseline", "process_sweep", "load_balance"}:
            continue
        totals = json.loads(row["rank_total_sec"]) if row["rank_total_sec"] else []
        computes = json.loads(row["rank_compute_sec"]) if row["rank_compute_sec"] else []
        comms = json.loads(row["rank_comm_sec"]) if row["rank_comm_sec"] else []
        local_rows = json.loads(row["local_rows_by_rank"]) if row["local_rows_by_rank"] else []
        for rank, total in enumerate(totals):
            compute = computes[rank] if rank < len(computes) else math.nan
            comm = comms[rank] if rank < len(comms) else math.nan
            out.append(
                {
                    "experiment": row["experiment"],
                    "size": row["size"],
                    "max_iters": row["max_iters"],
                    "np": row["np"],
                    "rep": row["rep"],
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
    return out


def write_markdown(out, baseline, process_sweep, input_calibration, workload_calibration, load_balance):
    lines = [
        "# C++ Sinkhorn Router3 Report Experiment Suite",
        "",
        "Cluster: `192.168.1.25/.26/.75` over router Ethernet. Hostfile uses the safe two-slot-per-node profile.",
        "",
        "Experiments covered:",
        "- Baseline size sweep: `np=3`, sizes `2000..12000`, fixed 50 iterations, 4 reps.",
        "- Process sweep: several `np` values up to the safe slots limit, fixed 50 iterations, 3 reps.",
        "- Input-size calibration: safe dense sizes up to `12000`, fixed 50 iterations, 1 rep.",
        "- Workload calibration: `np=3`, fixed size with longer iteration counts, 1 rep.",
        "- Load balance and compute/communication breakdown from per-rank metrics.",
        "",
        "Safety note: larger dense-size and 12-process attempts were stopped after node `.75` reported a kernel soft lockup. This suite caps memory pressure and uses longer iteration counts for the 2-3 minute workload target.",
        "",
        "## Baseline Size Sweep",
        "",
        "| size | seq avg s | mpi np3 avg s | speedup | efficiency | avg imbalance % | max obj diff |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline:
        lines.append(
            f"| {row['size']} | {fmt(row['seq_avg_sec'])} | {fmt(row['mpi_avg_sec'])} | "
            f"{fmt(row['speedup'])}x | {fmt(row['efficiency'])} | "
            f"{fmt(row['avg_rank_imbalance_pct'])} | {row['max_obj_diff']:.3e} |"
        )

    lines += [
        "",
        "## Process Sweep",
        "",
        "| np | mpi avg s | speedup vs seq | efficiency | avg imbalance % | max compute s | max comm s | max overhead s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in process_sweep:
        lines.append(
            f"| {row['np']} | {fmt(row['mpi_avg_sec'])} | {fmt(row['speedup_vs_seq'])}x | "
            f"{fmt(row['efficiency_vs_seq'])} | {fmt(row['avg_rank_imbalance_pct'])} | "
            f"{fmt(row['avg_max_rank_compute_sec'])} | {fmt(row['avg_max_rank_comm_sec'])} | "
            f"{fmt(row['avg_max_rank_overhead_sec'])} |"
        )

    lines += [
        "",
        "## Input-Size Calibration",
        "",
        "| size | max iters | np | mpi runtime s | avg imbalance % |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in input_calibration:
        lines.append(
            f"| {row['size']} | {row['max_iters']} | {row['np']} | "
            f"{fmt(row['mpi_avg_sec'])} | {fmt(row['avg_rank_imbalance_pct'])} |"
        )

    lines += [
        "",
        "## Workload Calibration",
        "",
        "| size | max iters | np | mpi runtime s | avg imbalance % |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in workload_calibration:
        lines.append(
            f"| {row['size']} | {row['max_iters']} | {row['np']} | "
            f"{fmt(row['mpi_avg_sec'])} | {fmt(row['avg_rank_imbalance_pct'])} |"
        )

    lines += [
        "",
        "## Load Balance Snapshot",
        "",
        "| experiment | size | max iters | np | count | runtime s | avg imbalance % | max compute s | max comm s | max overhead s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in load_balance:
        lines.append(
            f"| {row['experiment']} | {row['size']} | {row['max_iters']} | {row['np']} | "
            f"{row['count']} | {fmt(row['mpi_avg_sec'])} | {fmt(row['avg_rank_imbalance_pct'])} | "
            f"{fmt(row['avg_max_rank_compute_sec'])} | {fmt(row['avg_max_rank_comm_sec'])} | "
            f"{fmt(row['avg_max_rank_overhead_sec'])} |"
        )

    lines += [
        "",
        "Raw per-run data is in `per_run.csv`.",
        "Per-rank load-balance rows are in `load_balance_by_rank.csv`.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    out = Path(sys.argv[1])
    rows = read_rows(out)
    per_run_fields = [
        "experiment",
        "kind",
        "size",
        "np",
        "max_iters",
        "rep",
        "runtime_sec",
        "objective",
        "row_error",
        "col_error",
        "column_syncs",
        "column_payload_bytes",
        "hostnames",
        "local_rows_by_rank",
        "rank_compute_sec",
        "rank_comm_sec",
        "rank_total_sec",
        "max_rank_compute_sec",
        "max_rank_comm_sec",
        "max_rank_overhead_sec",
        "max_rank_total_sec",
        "min_rank_total_sec",
        "rank_imbalance_pct",
        "json",
    ]
    write_csv(out / "per_run.csv", sorted(rows, key=lambda r: (r["experiment"], r["size"], r["np"], r["max_iters"], r["rep"], r["kind"])), per_run_fields)

    baseline = build_baseline(rows)
    process_sweep = build_process_sweep(rows)
    input_calibration = build_simple_summary(rows, "input_calibration")
    workload_calibration = build_simple_summary(rows, "workload_calibration")
    load_balance = build_simple_summary(rows, "load_balance")
    compute_breakdown = build_simple_summary(rows, "baseline") + process_sweep + input_calibration + workload_calibration + load_balance
    load_balance_by_rank = build_load_balance_by_rank(rows)

    write_csv(out / "baseline_summary.csv", baseline, list(baseline[0].keys()) if baseline else ["size"])
    write_csv(out / "process_sweep_summary.csv", process_sweep, list(process_sweep[0].keys()) if process_sweep else ["np"])
    write_csv(out / "input_calibration_summary.csv", input_calibration, list(input_calibration[0].keys()) if input_calibration else ["size"])
    write_csv(out / "workload_calibration_summary.csv", workload_calibration, list(workload_calibration[0].keys()) if workload_calibration else ["max_iters"])
    write_csv(out / "load_balance_summary.csv", load_balance, list(load_balance[0].keys()) if load_balance else ["np"])
    write_csv(out / "compute_breakdown_summary.csv", compute_breakdown, union_fields(compute_breakdown) if compute_breakdown else ["experiment"])
    write_csv(out / "load_balance_by_rank.csv", load_balance_by_rank, list(load_balance_by_rank[0].keys()) if load_balance_by_rank else ["rank"])
    write_markdown(out, baseline, process_sweep, input_calibration, workload_calibration, load_balance)


if __name__ == "__main__":
    main()
