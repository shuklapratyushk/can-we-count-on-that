"""Executed hypothetical evidence sensitivity, not estimated biological priors."""
import json
from fractions import Fraction
from engine import ROOT

OUT = ROOT / "scale_results"


def main():
    OUT.mkdir(exist_ok=True)
    n, recorded, unknown = 512, 166, 110
    eta = Fraction(1, 40)
    # A single one-specimen-per-patient broad/cohort query is affine. Under a
    # cap c on unknown positives, every integer total from zero to c is feasible.
    caps = []
    for cap in (0, 10, 25, 26, 50, 110):
        lower, upper = Fraction(recorded, n), Fraction(recorded+cap, n)
        caps.append({"hypothetical_max_unknown_positives": cap, "lower": float(lower),
                     "upper": float(upper), "radius_pp": float(50*(upper-lower)),
                     "point_guard_passes": (upper-lower)/2 <= eta})
    required = next(t for t in range(unknown+1) if Fraction(unknown-t, 2*n) <= eta)
    assert required == 85
    assert next(row for row in caps if row["hypothetical_max_unknown_positives"] == 25)["point_guard_passes"]
    assert not next(row for row in caps if row["hypothetical_max_unknown_positives"] == 26)["point_guard_passes"]
    result = {"case": "KIRC-VHL patient/cohort/all/broad", "denominator": n,
              "recorded_positive": recorded, "unprofiled": unknown,
              "hypothetical_cap_results": caps,
              "additional_unknown_patients_to_measure_for_guaranteed_2_5pp_radius": required,
              "interpretation": "Caps illustrate conditional identification tightening and are not supported by this snapshot. Measuring 85 unknown patients guarantees at most 25 remaining unknown statuses, whatever their measured outcomes; assay error remains outside the model.",
              "cross_patient_constraint_note": "A global positive-count cap couples blocks. The unrestricted separable support theorem cannot be reused for all paired constrained queries without a constrained optimizer."}
    (OUT / "constraint_sensitivity.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
