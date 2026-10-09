# CLARA standalone reproducibility artifact

This is the current shared-completion research artifact.

## Reproduce

Use Python 3.12 and the pinned requirements:

```bash
python -m pip install -r requirements.txt
python reproducibility/run_all.py
python reproducibility/guard_demo.py --demo
python reproducibility/guard_demo.py --verify scale_results/guard_receipt_2.json
```

The runner is offline and uses the bundled snapshots. It reproduces exact
counting/certification results, hypothetical policy scenarios, synthetic
verification and timing workloads. Runtime measurements will vary by host.
Forty-three contracts passed in the retained execution, including nine actual typed-guard contracts.

## Sources and attribution

Frozen TCGA/cBioPortal bytes and retrieval hashes: data/snapshot_manifest.json.
MSK-IMPACT data are pinned to DataHub commit
dca75cb3f32b82d54a6f78bf0a6323e5b975aca1; source files match Git LFS hashes.
Preserve data/msk/LICENSE (ODbL) and Zehir et al., Nature Medicine 2017,
DOI 10.1038/nm.4333. Source and adapted MSK data retain that source license.
IEEEtran retains its existing license and copyright. The authors must select
their code licensing terms before a public open-source release; no additional
license is silently imposed here.

SHA-256 receipt/manifest values support integrity and replay, not signatures,
authorship verification, clinical truth, or independently replicated biology.
Only supplied typed candidates and the declared completion model are certified.
