"""A typed agent-facing guard and hash-bound provenance receipt.

No language model is invoked. The upstream caller supplies supported typed
fields. SHA-256 detects changed local content, not authorship or trustworthiness.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

from engine import ROOT
from audit_query import make_audit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bindings():
    manifest = json.loads((ROOT / "data/snapshot_manifest.json").read_text())
    files = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError("Frozen snapshot file changed: " + entry["path"])
        files.append({"path": entry["path"], "sha256": actual})
    return {"snapshot_content_sha256": digest(files),
            "protocol_sha256": hashlib.sha256((ROOT / "reproducibility/protocol.json").read_bytes()).hexdigest(),
            "completion_protocol_sha256": hashlib.sha256((ROOT / "reproducibility/completion_protocol.json").read_bytes()).hexdigest(),
            "implementation_sha256": digest([{ "path": name, "sha256": hashlib.sha256((ROOT / "reproducibility" / name).read_bytes()).hexdigest()}
                for name in ("engine.py", "completion.py", "audit_query.py", "guard_demo.py")]),
            "completion_model": "independent patient blocks; shared nested specimen event states; profiled recorded statuses fixed",
            "provenance_scope": "Query-level result and execution lineage; not tuple-level why/where provenance or a digital signature"}


def guard(request):
    allowed = {"cohort", "gene", "fields", "point_error_budget_pp"}
    if set(request)-allowed:
        raise ValueError("Unsupported request attribute")
    if not {"cohort", "gene"} <= set(request):
        raise ValueError("Cohort and gene required")
    source = bindings()
    card = make_audit(request["cohort"].lower(), request["gene"].upper(), request.get("fields", {}),
                      request.get("point_error_budget_pp", 2.5))
    receipt = {"schema": "clara.guard-receipt.v1", "request": request, "bindings": source,
               "certificate": card, "request_sha256": digest(request), "certificate_sha256": digest(card)}
    source_id = "urn:sha256:" + source["snapshot_content_sha256"]
    activity_id = "urn:sha256:" + digest({"request": request, "bindings": source})
    receipt["provenance"] = {"@context": {"prov": "http://www.w3.org/ns/prov#"},
        "@graph": [{"@id": source_id, "@type": "prov:Entity"},
                   {"@id": activity_id, "@type": "prov:Activity", "prov:used": [{"@id": source_id}]},
                   {"@id": "urn:sha256:" + receipt["certificate_sha256"], "@type": "prov:Entity",
                    "prov:wasGeneratedBy": {"@id": activity_id}}]}
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def verify(receipt, replay=False):
    content = copy.deepcopy(receipt)
    expected = content.pop("receipt_sha256")
    if digest(content) != expected:
        raise ValueError("Receipt content hash mismatch")
    if digest(content["request"]) != content["request_sha256"] or digest(content["certificate"]) != content["certificate_sha256"]:
        raise ValueError("Request or certificate hash mismatch")
    if content["bindings"] != bindings():
        raise ValueError("Snapshot, protocol, or implementation binding changed")
    if replay:
        regenerated = guard(content["request"])
        if regenerated["receipt_sha256"] != expected:
            raise ValueError("Replayed certificate differs")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(json.loads(args.verify.read_text()), replay=True)
        print("Hash bindings and full replay verified.")
    elif args.request:
        print(json.dumps(guard(json.loads(args.request.read_text())), indent=2))
    elif args.demo:
        out = ROOT / "scale_results"
        out.mkdir(exist_ok=True)
        requests = [{"cohort": "kirc", "gene": "VHL", "fields": {}},
                    {"cohort": "kirc", "gene": "VHL", "fields": {"assay_denominator": "cohort", "sample_context": "all", "mutation_scope": "broad"}},
                    {"cohort": "kirc", "gene": "VHL", "fields": {"analysis_unit": "patient", "assay_denominator": "profiled", "sample_context": "all", "mutation_scope": "broad"}}]
        cards = []
        for index, request in enumerate(requests, 1):
            receipt = guard(request)
            verify(receipt, replay=True)
            (out / f"guard_receipt_{index}.json").write_text(json.dumps(receipt, indent=2)+"\n")
            cards.append({"request": request, "status": receipt["certificate"]["status"],
                          "answer": receipt["certificate"]["answer"], "replay_verified": True})
        (out / "guard_demo_summary.json").write_text(json.dumps(cards, indent=2)+"\n")
        print(json.dumps(cards, indent=2))
    else:
        parser.error("Supply --request, --verify, or --demo")


if __name__ == "__main__":
    main()
