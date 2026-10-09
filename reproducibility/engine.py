"""Two counting implementations and a result-aware clarification planner.

The planner operates on supplied typed interpretations. This module does not
claim to reproduce the original natural-language parser.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "reproducibility/protocol.json").read_text())
FIELDS = PROTOCOL["fields"]
VALUES = [PROTOCOL["field_values"][field] for field in FIELDS]
ASSIGNMENTS = list(product((0, 1), repeat=len(FIELDS)))


@dataclass(frozen=True)
class Spec:
    cohort: str
    gene: str
    analysis_unit: str = "patient"
    assay_denominator: str = "profiled"
    sample_context: str = "all"
    mutation_scope: str = "broad"

    def __post_init__(self):
        for field, allowed in zip(FIELDS, VALUES):
            if getattr(self, field) not in allowed:
                raise ValueError(f"Unsupported {field}: {getattr(self, field)}")

    @classmethod
    def from_bits(cls, cohort, gene, bits):
        return cls(cohort, gene, **{field: allowed[bit]
                                  for field, allowed, bit in zip(FIELDS, VALUES, bits)})


@dataclass(frozen=True)
class Result:
    positive: int
    denominator: int
    profiled_units: int

    def __post_init__(self):
        if not 0 <= self.positive <= self.profiled_units <= self.denominator:
            raise ValueError("Numerator/eligibility invariant violated")

    @property
    def prevalence(self):
        return self.positive / self.denominator if self.denominator else None

    @property
    def missingness_upper_bound(self):
        if not self.denominator:
            return None
        return (self.positive + self.denominator - self.profiled_units) / self.denominator

    def to_dict(self):
        return {**asdict(self), "prevalence": self.prevalence,
                "missingness_upper_bound": self.missingness_upper_bound}


def load_snapshot():
    samples = []
    mutations = []
    for code in PROTOCOL["original_cohorts"] + PROTOCOL["additional_cohorts"]:
        directory = ROOT / "data/raw" / code
        raw_samples = json.loads((directory / "samples.json").read_text())
        profiled = set(json.loads((directory / "sequenced.json").read_text())["sampleIds"])
        for row in raw_samples:
            samples.append({"cohort": code, "sample": row["sampleId"],
                            "patient": row["patientId"],
                            "primary": row["sampleType"] == PROTOCOL["primary_sample_type"],
                            "sample_type": row["sampleType"],
                            "profiled": row["sampleId"] in profiled})
        for row in json.loads((directory / "mutations.json").read_text()):
            mutations.append({"cohort": code, "sample": row["sampleId"],
                              "gene": row["gene"]["hugoGeneSymbol"],
                              "mutation_type": row["mutationType"]})
    return pd.DataFrame(samples), pd.DataFrame(mutations)


class PandasEngine:
    """Filter eligible specimens before projecting to patient/sample sets."""
    def __init__(self, samples, mutations):
        self.samples = samples.copy()
        self.mutations = mutations.copy()
        self.cohorts = {key: value for key, value in self.samples.groupby("cohort")}
        self.gene_mutations = {key: value for key, value in self.mutations.groupby(["cohort", "gene"])}

    def execute(self, spec):
        context = self.cohorts.get(spec.cohort, self.samples.iloc[:0])
        if spec.sample_context == "primary":
            context = context[context["primary"].astype(bool)]
        profiled = context[context["profiled"].astype(bool)]
        eligible = profiled if spec.assay_denominator == "profiled" else context
        unit = spec.analysis_unit
        denominator_ids = set(eligible[unit])
        profiled_ids = set(profiled[unit])
        consequence_set = set(PROTOCOL[spec.mutation_scope + "_types"])
        records = self.gene_mutations.get((spec.cohort, spec.gene), self.mutations.iloc[:0])
        positive_rows = records[records["mutation_type"].isin(consequence_set) &
                                records["sample"].isin(set(profiled["sample"]))]
        positive_samples = set(positive_rows["sample"])
        positive_ids = set(profiled.loc[profiled["sample"].isin(positive_samples), unit])
        return Result(len(positive_ids & denominator_ids), len(denominator_ids),
                      len(profiled_ids & denominator_ids))


class SQLiteEngine:
    """A separately expressed SQL COUNT(DISTINCT) implementation."""
    def __init__(self, samples, mutations):
        self.connection = sqlite3.connect(":memory:")
        samples.to_sql("samples", self.connection, index=False)
        mutations.to_sql("mutations", self.connection, index=False)
        self.connection.executescript("""
            CREATE INDEX sample_lookup ON samples(cohort, sample);
            CREATE INDEX mutation_lookup ON mutations(cohort, gene, sample);
        """)

    def execute(self, spec):
        unit = {"patient": "patient", "sample": "sample"}[spec.analysis_unit]
        consequence_types = PROTOCOL[spec.mutation_scope + "_types"]
        placeholders = ",".join("?" for _ in consequence_types)
        context_clause = 'AND "primary" = 1' if spec.sample_context == "primary" else ""
        eligibility_clause = "AND profiled = 1" if spec.assay_denominator == "profiled" else ""
        positive_context_clause = 'AND s."primary" = 1' if spec.sample_context == "primary" else ""
        sql = f"""
          SELECT
            (SELECT COUNT(DISTINCT s.{unit})
             FROM samples s INNER JOIN mutations m
               ON s.cohort = m.cohort AND s.sample = m.sample
             WHERE s.cohort = ? AND s.profiled = 1 {positive_context_clause}
               AND m.gene = ? AND m.mutation_type IN ({placeholders})),
            (SELECT COUNT(DISTINCT {unit}) FROM samples
             WHERE cohort = ? {context_clause} {eligibility_clause}),
            (SELECT COUNT(DISTINCT {unit}) FROM samples
             WHERE cohort = ? {context_clause} AND profiled = 1)
        """
        # Numerators use only profiled specimens within the same context, so
        # their projected unit sets are subsets of either allowed denominator.
        counts = self.connection.execute(sql, (spec.cohort, spec.gene, *consequence_types,
                                              spec.cohort, spec.cohort)).fetchone()
        return Result(*(int(value) for value in counts))


def divergence(a, b):
    if a is None or b is None:
        return None, None
    delta = abs(a - b)
    return 100.0 * delta, delta / ((abs(a) + abs(b)) / 2.0 + 1e-12)


def sensitive(a, b, relative=0.10, absolute_pp=5.0):
    absolute, ratio = divergence(a, b)
    if absolute is None:
        return True  # An undefined result cannot certify a direct answer.
    return ratio > relative or absolute > absolute_pp


def stable(values, relative=0.10, absolute_pp=5.0):
    if not values or any(value is None for value in values):
        return False
    return not sensitive(min(values), max(values), relative, absolute_pp)


class ClarificationPlanner:
    """Minimum expected question count under an explicit uniform intent prior.

    Optimality is conditional on a complete supplied candidate set, truthful
    field responses, unit question costs, and the specified stopping tolerance.
    It is not a claim about the original parser or observed user behavior.
    """
    def __init__(self, values, assignments=ASSIGNMENTS):
        if len(values) != len(assignments):
            raise ValueError("Candidate/value length mismatch")
        self.values = tuple(values)
        self.assignments = tuple(tuple(a) for a in assignments)
        self.root = tuple(range(len(values)))

    def stable(self, state):
        return stable([self.values[i] for i in state])

    def partition(self, state, field):
        return tuple(tuple(i for i in state if self.assignments[i][field] == bit)
                     for bit in (0, 1))

    @lru_cache(maxsize=None)
    def solve(self, state):
        if self.stable(state) or len(state) <= 1:
            return 0.0, None
        choices = []
        for field in range(len(self.assignments[0])):
            groups = self.partition(state, field)
            if any(not group for group in groups):
                continue
            expectation = 1.0 + sum(len(group) / len(state) * self.solve(group)[0]
                                    for group in groups)
            choices.append((expectation, field))
        return min(choices)

    def walk(self, truth_index, order=None):
        state = self.root
        questions = []
        while not self.stable(state) and len(state) > 1:
            if order is None:
                field = self.solve(state)[1]
            else:
                field = next((f for f in order if len({self.assignments[i][f] for i in state}) > 1), None)
            if field is None:
                raise RuntimeError("No field can separate remaining candidates")
            truth_value = self.assignments[truth_index][field]
            questions.append(field)
            state = tuple(i for i in state if self.assignments[i][field] == truth_value)
        assert truth_index in state
        if len(state) > 1:
            assert self.stable(state)
        return len(questions), questions, state
