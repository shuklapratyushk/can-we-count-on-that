"""Reproduce the standalone frozen-snapshot counting and observed-query registry."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

from engine import (ROOT, PROTOCOL, FIELDS, ASSIGNMENTS, Spec, PandasEngine,
                    SQLiteEngine, divergence, sensitive, stable, ClarificationPlanner,
                    load_snapshot)


def verify_snapshot():
    manifest = json.loads((ROOT/"data/snapshot_manifest.json").read_text())
    assert not manifest["errors"], "Snapshot has failed cohorts"
    assert set(manifest["cohorts_requested"]) == set(PROTOCOL["original_cohorts"]+PROTOCOL["additional_cohorts"])
    for record in manifest["files"]:
        data = (ROOT/record["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
    protocol_hash = hashlib.sha256((ROOT/"reproducibility/protocol.json").read_bytes()).hexdigest()
    assert protocol_hash == manifest["protocol_sha256_before_download"], "Protocol changed after snapshot start"
    return manifest


def main():
    started = time.perf_counter()
    manifest = verify_snapshot()
    samples, mutations = load_snapshot()
    samples.to_csv(ROOT/"data/normalized/samples.csv",index=False)
    mutations.to_csv(ROOT/"data/normalized/mutations.csv",index=False)
    mutations.groupby("mutation_type").size().rename("records").to_csv(ROOT/"results/mutation_type_counts.csv")
    s_engine, p_engine = SQLiteEngine(samples,mutations), PandasEngine(samples,mutations)
    interpretations, contrasts, families, plans, dialogue = [], [], [], [], []
    cohort_rows = []
    max_disagreement = 0.0
    all_orders = list(permutations(range(4)))
    codes = PROTOCOL["original_cohorts"]+PROTOCOL["additional_cohorts"]
    for code in codes:
        subset = samples[samples.cohort.eq(code)]
        cohort_rows.append({"cohort":code.upper(),"group":"original" if code in PROTOCOL["original_cohorts"] else "additional",
                            "samples":len(subset),"patients":subset.patient.nunique(),
                            "profiled_samples":int(subset.profiled.sum()),
                            "profiled_patients":subset.loc[subset.profiled,"patient"].nunique(),
                            "primary_samples":int(subset.primary.sum()),
                            "primary_profiled_samples":int((subset.primary & subset.profiled).sum()),
                            "panel_mutation_records":int(mutations.cohort.eq(code).sum())})
        for gene in PROTOCOL["genes"]:
            group = "original" if code in PROTOCOL["original_cohorts"] else "additional"
            results = []
            for index,bits in enumerate(ASSIGNMENTS):
                spec = Spec.from_bits(code,gene,bits)
                sql_result, pandas_result = s_engine.execute(spec), p_engine.execute(spec)
                assert sql_result == pandas_result, (spec,sql_result,pandas_result)
                results.append(sql_result)
                interpretations.append({"interpretation_id":f"{code}:{gene}:{index:02d}",
                                       "group":group,"index":index,**asdict(spec),
                                       **sql_result.to_dict(),"engines_match":True})
            values = [row.prevalence for row in results]
            # For each family: reference vs one binary field flip. Including
            # zero-mutation families prevents result-dependent selection bias.
            sensitive_axes = []
            for f,field in enumerate(FIELDS):
                alternative_index = 1 << (3-f)
                a,b = values[0],values[alternative_index]
                absolute,relative = divergence(a,b)
                label = sensitive(a,b)
                sensitive_axes.append(label)
                contrasts.append({"contrast_id":f"{code}:{gene}:{field}","group":group,
                                  "cohort":code.upper(),"gene":gene,"field":field,
                                  "p_reference":a,"p_alternative":b,
                                  "reference_positive":results[0].positive,
                                  "alternative_positive":results[alternative_index].positive,
                                  "absolute_pp":absolute,"relative":relative,"sensitive":label})
            broad_indices = [i for i,bits in enumerate(ASSIGNMENTS) if bits[3] == 0]
            broad_values = [values[i] for i in broad_indices]
            broad_sensitive = not stable(broad_values)
            finite = [v for v in values if v is not None]
            lo,hi = min(finite),max(finite)
            full_abs,full_rel = divergence(lo,hi)
            broad_abs,broad_rel = divergence(min(broad_values),max(broad_values))
            # Check pairwise equivalence independently of the extrema decision.
            pair_flags = [sensitive(values[i],values[j]) for i in range(16) for j in range(i+1,16)]
            assert any(pair_flags) == (not stable(values))
            planner = ClarificationPlanner(values)
            optimum,_ = planner.solve(planner.root)
            optimal_lengths = []
            terminal_sizes = []
            for truth_index in range(16):
                length,questions,state = planner.walk(truth_index)
                optimal_lengths.append(length)
                terminal_sizes.append(len(state))
                terminal_values = [values[i] for i in state]
                dialogue.append({"cohort":code.upper(),"gene":gene,"intent_index":truth_index,
                                 "questions":length,"field_sequence":";".join(FIELDS[f] for f in questions),
                                 "remaining_interpretations":len(state),
                                 "terminal_min":min(terminal_values),"terminal_max":max(terminal_values),
                                 "truth_prevalence":values[truth_index],"terminal_stable":stable(terminal_values)})
            assert abs(sum(optimal_lengths)/16-optimum) < 1e-12
            fixed_means = []
            for order in all_orders:
                lengths = [planner.walk(i,order)[0] for i in range(16)]
                mean = float(np.mean(lengths))
                fixed_means.append(mean)
                assert optimum <= mean+1e-12
                plans.append({"cohort":code.upper(),"gene":gene,"group":group,
                              "order":";".join(FIELDS[f] for f in order),
                              "mean_questions":mean,"max_questions":max(lengths)})
            cohort_family_denominator = results[4]  # patient/cohort/all/broad
            families.append({"cohort":code.upper(),"gene":gene,"group":group,
                             "full_min":lo,"full_max":hi,"full_absolute_pp":full_abs,"full_relative":full_rel,
                             "full_sensitive":not stable(values),
                             "any_one_axis_sensitive":any(sensitive_axes),
                             "joint_sensitive_one_axis_stable":not stable(values) and not any(sensitive_axes),
                             "broad_sensitive":broad_sensitive,"broad_absolute_pp":broad_abs,"broad_relative":broad_rel,
                             "broad_joint_sensitive_one_axis_stable":broad_sensitive and not any(sensitive_axes[:3]),
                             "reference_prevalence":values[0],"reference_positive":results[0].positive,
                             "optimal_mean_questions":optimum,"optimal_max_questions":max(optimal_lengths),
                             "mean_terminal_candidates":float(np.mean(terminal_sizes)),
                             "best_family_fixed_order":min(fixed_means),"random_fixed_order_mean":float(np.mean(fixed_means)),
                             "cohort_missingness_width_pp":100*(cohort_family_denominator.missingness_upper_bound-cohort_family_denominator.prevalence),
                             "cohort_observed_fraction":cohort_family_denominator.prevalence,
                             "cohort_possible_upper":cohort_family_denominator.missingness_upper_bound})
        print(f"Verified {code.upper()}: {len(PROTOCOL['genes'])*16} interpretations and all clarification paths",flush=True)
    spec_df,contrast_df,family_df = pd.DataFrame(interpretations),pd.DataFrame(contrasts),pd.DataFrame(families)
    fixed_df,dialogue_df,cohort_df = pd.DataFrame(plans),pd.DataFrame(dialogue),pd.DataFrame(cohort_rows)
    for name,frame in [("interpretations",spec_df),("single_axis_contrasts",contrast_df),
                       ("families",family_df),("fixed_order_plans",fixed_df),
                       ("simulated_clarification_paths",dialogue_df),("cohort_summary",cohort_df)]:
        frame.to_csv(ROOT/f"results/{name}.csv",index=False)
    # Equal family weights; hypothetical replies are not independent observations.
    fixed_summary = fixed_df.groupby("order")["mean_questions"].mean().sort_values()
    fixed_summary.to_csv(ROOT/"results/fixed_order_summary.csv")
    mechanism_summary = contrast_df.groupby(["group","field"])["sensitive"].agg(["sum","count","mean"])
    mechanism_summary.to_csv(ROOT/"results/mechanism_summary.csv")
    threshold_rows = []
    for relative in PROTOCOL["sensitivity_tolerances"]["relative"]:
        for absolute in PROTOCOL["sensitivity_tolerances"]["absolute_percentage_points"]:
            for group in ("all","original","additional"):
                f = family_df if group == "all" else family_df[family_df.group.eq(group)]
                c = contrast_df if group == "all" else contrast_df[contrast_df.group.eq(group)]
                threshold_rows.append({"relative_threshold":relative,"absolute_pp_threshold":absolute,"group":group,
                                       "full_sensitive":int(((f.full_relative>relative)|(f.full_absolute_pp>absolute)).sum()),
                                       "families":len(f),
                                       "single_axis_sensitive":int(((c.relative>relative)|(c.absolute_pp>absolute)).sum()),
                                       "single_axis_contrasts":len(c)})
    pd.DataFrame(threshold_rows).to_csv(ROOT/"results/threshold_sensitivity.csv",index=False)
    # Descriptive cluster resampling of the finite cohort panel; these are not
    # calibrated guarantees for new institutions or user populations.
    rng = np.random.default_rng(20261006)
    by_cohort = family_df.groupby("cohort").optimal_mean_questions.mean()
    bootstrap_means = np.mean(rng.choice(by_cohort.values,size=(10000,len(by_cohort)),replace=True),axis=1)
    summary = {"created_at_utc":datetime.now(timezone.utc).isoformat(),
               "new_results_provenance":"Frozen public cBioPortal snapshot; complete observed-query registry",
               "samples":int(len(samples)),"patients":int(samples[["cohort","patient"]].drop_duplicates().shape[0]),
               "profiled_samples":int(samples.profiled.sum()),"panel_mutation_records":int(len(mutations)),
               "cohorts":len(codes),"genes":len(PROTOCOL["genes"]),"families":len(family_df),
               "interpretations":len(spec_df),"cross_engine_matches":int(spec_df.engines_match.sum()),
               "single_axis_contrasts":len(contrast_df),"single_axis_sensitive":int(contrast_df.sensitive.sum()),
               "full_sensitive_families":int(family_df.full_sensitive.sum()),
               "full_sensitive_one_axis_stable":int(family_df.joint_sensitive_one_axis_stable.sum()),
               "broad_sensitive_families":int(family_df.broad_sensitive.sum()),
               "broad_sensitive_one_axis_stable":int(family_df.broad_joint_sensitive_one_axis_stable.sum()),
               "optimal_mean_questions":float(family_df.optimal_mean_questions.mean()),
               "best_global_fixed_order":fixed_summary.index[0],
               "best_global_fixed_order_mean":float(fixed_summary.iloc[0]),
               "random_fixed_order_mean":float(fixed_df.mean_questions.mean()),
               "always_ask_four_mean":4.0,"optimal_max_questions":int(family_df.optimal_max_questions.max()),
               "saving_vs_always_four":1-float(family_df.optimal_mean_questions.mean())/4,
               "saving_vs_best_global_fixed_order":1-float(family_df.optimal_mean_questions.mean())/float(fixed_summary.iloc[0]),
               "simulation_paths":len(dialogue_df),"simulation_all_terminal_states_stable":bool(dialogue_df.terminal_stable.all()),
               "descriptive_cohort_bootstrap_mean_questions_95":np.quantile(bootstrap_means,[.025,.975]).tolist(),
               "bootstrap_scope":"Descriptive resampling of 12 fixed cohorts, 30 genes retained per cohort; not a deployment-confidence interval",
               "runtime_seconds":time.perf_counter()-started,
               "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__},
               "snapshot_started_at_utc":manifest["snapshot_started_at_utc"],
               "snapshot_completed_at_utc":manifest["snapshot_completed_at_utc"]}
    (ROOT/"results/summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({key:value for key,value in summary.items() if key not in ["runtime"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
