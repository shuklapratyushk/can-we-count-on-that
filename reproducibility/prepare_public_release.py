"""Prepare a standalone publication artifact; do not publish externally."""
from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'deliverables'
PUBLIC_DIRS={'data','results','completion_results','msk_results','scale_results','figures_next','manuscript','latex_submission','reproducibility'}
EXCLUDED_NAMES={'legacy_aggregate_audit.csv','cost_break_even.csv','legacy_table.tex','public_artifact_check.json','latex_isolated_main_log.txt','latex_isolated_supplement_log.txt','latex_isolated_poster_abstract_log.txt'}
EXCLUDED_SUFFIXES={'.aux','.log','.out','.fls','.fdb_latexmk','.pyc','.tmp'}


def main():
    OUT.mkdir(exist_ok=True)
    paths=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(ROOT)
        if rel.parts[0] not in PUBLIC_DIRS and rel.as_posix() not in {'requirements.txt','CLARA_reproduce.ipynb'}:continue
        if any(part.startswith('.') or part=='__pycache__' for part in rel.parts):continue
        if path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:continue
        if rel.parts[0]=='manuscript' and (path.suffix=='.txt' or path.name=='reference_appendix.tex'):continue
        if rel.parts[0]=='reproducibility' and path.name in {'download_public.py','download_msk.py','make_figures.py','package_release.py','build_deliverables.py'}:continue
        paths.append(path)
    readme='''# CLARA standalone reproducibility artifact

This is the current shared-completion research artifact. The IEEE main paper
has eight pages including nineteen references. It is a prepared source/data
release, without an assigned public repository URL, archival DOI, or external
publication. Original rejected-paper material and historical pilot aggregates
are excluded.

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
No human or LLM evaluation is asserted. Forty-three contracts passed in the
retained execution, including nine actual typed-guard contracts.

For Overleaf, upload the contents of latex_submission and choose main.tex.
The unmodified IEEEtran class and all vector figure dependencies are included;
no bibliography run is needed. The proof supplement and abstract are separate
author materials. The current documents disclose OpenAI ChatGPT (Codex)
assistance throughout drafting, mathematics, software and figure design.

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
'''
    manifest={'artifact':'CLARA standalone shared-completion reproducibility release','version':'2026.10.07',
              'public_url':None,'doi':None,'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    citation='''cff-version: 1.2.0
message: "Cite the standalone CLARA research artifact; this prepared release has no assigned DOI."
type: software
title: "CLARA: Shared-Completion Certificates for Cancer-Genomics Query Clarification"
version: "2026.10.07"
date-released: "2026-10-07"
authors:
  - family-names: Shukla
    given-names: Pratyush Kumar
  - family-names: Tib
    given-names: Manveer Singh
  - family-names: Garg
    given-names: Siddhant
'''
    archive=OUT/'CLARA_Reproducibility_Release.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        bundle.writestr('README.md',readme)
        bundle.writestr('CITATION.cff',citation)
        bundle.writestr('ARTIFACT_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        for path in paths:bundle.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for entry in manifest['files']:
            data=bundle.read(entry['path'])
            assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
        assert not any('original_submission' in name or 'legacy_aggregate' in name for name in bundle.namelist())
    (ROOT/'scale_results/public_artifact_check.json').write_text(json.dumps({'status':'passed','files':len(paths)+3,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'external_publication':False},indent=2)+'\n')
    print(f'Standalone reproducibility archive prepared: {len(paths)+3} files, {archive.stat().st_size:,} bytes.')


if __name__=='__main__':main()
