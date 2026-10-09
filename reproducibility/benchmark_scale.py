"""Measured serial certificate scaling; synthetic rows are not new patients."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from engine import ROOT, PROTOCOL, ASSIGNMENTS, Result, load_snapshot
from completion import CompletionModel, GraphPlanner

OUT = ROOT / "scale_results"
PROTOCOL_PATH = ROOT / "reproducibility/scale_protocol.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_protocol():
    OUT.mkdir(exist_ok=True)
    record = OUT / "protocol_record.json"
    if record.exists():
        assert json.loads(record.read_text())["protocol_sha256"] == sha(PROTOCOL_PATH)
    else:
        record.write_text(json.dumps({"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "protocol_sha256": sha(PROTOCOL_PATH),
                                     "status": "Recorded locally before new measurements; not public preregistration"}, indent=2) + "\n")


def measure(fn, repetitions, warmups=1):
    for _ in range(warmups):
        fn()
    rows = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        fn()
        rows.append((time.perf_counter_ns() - start) / 1e6)
    return rows


def summary(times):
    return {"median_ms": float(np.median(times)), "min_ms": min(times), "max_ms": max(times)}


def audit_kernel(model):
    graph = model.certificate_graph()
    model.absolute_certificate()
    model.relative_certificate(iterations=40)
    planner = GraphPlanner(graph)
    return planner.solve(planner.root)[0]


def local_table(m):
    # One patient, all m unknown specimens primary. Count/OR and nested classes
    # exactly match the specimen/patient cohort semantics. A profiled specimen
    # with no event supplies a positive denominator in profiled interpretations.
    states = np.asarray(list(product((0, 1, 2), repeat=m)), dtype=np.int8)
    broad = (states >= 1).sum(axis=1)
    restricted = (states == 2).sum(axis=1)
    table = np.zeros((len(states), 16), dtype=np.int64)
    denominators = np.ones(16, dtype=np.int64)
    for index, (unit, assay, context, scope) in enumerate(ASSIGNMENTS):
        positive = broad if scope == 0 else restricted
        if assay == 1:
            table[:, index] = positive if unit == 1 else positive > 0
            denominators[index] = m + 1 if unit == 1 else 1
    return table, denominators


def bare_model(table, denominators, multiplicity=1):
    model = CompletionModel.__new__(CompletionModel)
    model.denominators = denominators
    model.fixed_counts = np.zeros(16, dtype=np.int64)
    model._grouped = [(table, multiplicity)]
    return model


def main():
    record_protocol()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    repetitions = protocol["repetitions"]
    samples, mutations = load_snapshot()
    baseline = pd.read_csv(ROOT / "results/interpretations.csv")
    cohorts = PROTOCOL["original_cohorts"] + PROTOCOL["additional_cohorts"]
    genes = PROTOCOL["genes"]
    models = {}
    construction = []
    for cohort in cohorts:
        start = time.perf_counter_ns()
        for gene in genes:
            rows = baseline.loc[baseline.cohort.eq(cohort) & baseline.gene.eq(gene)].sort_values("index")
            observed = [Result(int(r.positive), int(r.denominator), int(r.profiled_units)) for _, r in rows.iterrows()]
            models[(cohort, gene)] = CompletionModel(samples, mutations, cohort, gene, observed)
        construction.append({"cohort": cohort, "genes": len(genes), "model_construction_ms": (time.perf_counter_ns()-start)/1e6})
        print("Built local tables for " + cohort, flush=True)
    pd.DataFrame(construction).to_csv(OUT / "model_construction.csv", index=False)
    jobs = [("cohorts", c, 30) for c in protocol["registry_cohort_counts"]]
    jobs += [("genes", 12, g) for g in protocol["registry_gene_counts"]]
    rng = np.random.default_rng(protocol["seed"])
    registry = []
    # Randomize configuration order to reduce a systematic warmup/order effect.
    for axis, c, g in [jobs[i] for i in rng.permutation(len(jobs))]:
        chosen = [models[(cohort, gene)] for cohort in cohorts[:c] for gene in genes[:g]]
        def run():
            return sum(audit_kernel(model) for model in chosen)
        times = measure(run, repetitions, protocol["warmups"])
        registry.append({"axis": axis, "cohorts": c, "genes": g, "families": c*g,
                         "local_states": sum(m.local_state_count for m in chosen),
                         "distinct_tables": sum(len(m._grouped) for m in chosen),
                         "times_ms": json.dumps(times), **summary(times)})
        print(f"Timed registry {c} cohorts x {g} genes: {np.median(times):.1f} ms", flush=True)
    pd.DataFrame(registry).to_csv(OUT / "registry_scaling.csv", index=False)
    table, denominators = local_table(1)
    blocks = []
    for count in protocol["synthetic_patient_blocks"]:
        grouped = bare_model(table, denominators * count, count)
        ungrouped = copy.copy(grouped)
        ungrouped._grouped = [(table, 1)] * count
        np.testing.assert_array_equal(grouped.support_numerator(19, -21), ungrouped.support_numerator(19, -21))
        for label, model in [("grouped", grouped), ("ungrouped", ungrouped)]:
            times = measure(lambda: model.support_numerator(19, -21), repetitions, protocol["warmups"])
            blocks.append({"blocks": count, "local_states": 3*count, "distinct_tables": 1,
                           "implementation": label, "times_ms": json.dumps(times), **summary(times)})
        print("Timed synthetic block count " + str(count), flush=True)
    pd.DataFrame(blocks).to_csv(OUT / "block_scaling.csv", index=False)
    local = []
    for m in protocol["local_unknown_specimens"]:
        start = time.perf_counter_ns()
        table, denominators = local_table(m)
        build_ms = (time.perf_counter_ns()-start)/1e6
        model = bare_model(table, denominators)
        times = measure(lambda: model.support_numerator(19, -21), repetitions, protocol["warmups"])
        local.append({"unknown_specimens": m, "states": 3**m, "table_bytes": table.nbytes,
                      "table_construction_ms": build_ms, "times_ms": json.dumps(times), **summary(times)})
    pd.DataFrame(local).to_csv(OUT / "local_block_scaling.csv", index=False)
    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith("model name")), cpu)
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(PROTOCOL_PATH), "python": platform.python_version(),
              "numpy": np.__version__, "platform": platform.platform(), "cpu": cpu,
              "visible_logical_cpus": os.cpu_count(), "workers": 1,
              "blas_thread_env": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "model_construction_total_ms": sum(row["model_construction_ms"] for row in construction),
              "registry": registry, "blocks": blocks, "local_blocks": local,
              "validation": "Grouped and ungrouped integer supports agree exactly at every synthetic block count",
              "limitations": protocol["interpretation"]}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Serial scaling benchmark completed.", flush=True)


if __name__ == "__main__":
    main()
