"""Make complete, flat LaTeX sources with inline IEEE references for Overleaf."""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
SOURCE = ROOT / "latex_submission"
OUT = ROOT.parent / "deliverables"
INPUT = re.compile(r"\\input\{([^}]+)\}")


def expand(path: Path, stack: tuple[Path, ...] = ()) -> str:
    path = path.resolve()
    if path in stack or path.parent != MANUSCRIPT.resolve():
        raise ValueError(f"Unexpected or recursive TeX input: {path.name}")
    text = path.read_text()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        child = MANUSCRIPT / (name if Path(name).suffix else name + ".tex")
        return "\n% Begin inline source: " + child.name + "\n" + expand(child, stack + (path,)) + "\n% End inline source: " + child.name + "\n"

    text = INPUT.sub(replace, text)
    text = text.replace(r"\graphicspath{{../figures_next/}}", r"\graphicspath{{figures/}}")
    return text.replace(r"\graphicspath{{../figures_next/}{../figures/}}", r"\graphicspath{{figures/}}")


def build_sources() -> None:
    SOURCE.mkdir(exist_ok=True)
    (SOURCE / "figures").mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    for original, name in [("paper.tex", "main.tex"), ("supplement.tex", "supplement.tex"), ("poster_abstract.tex", "poster_abstract.tex")]:
        content = expand(MANUSCRIPT / original)
        assert not INPUT.search(content)
        (SOURCE / name).write_text(content)
    main = (SOURCE / "main.tex").read_text()
    assert main.count(r"\bibitem{") == 19
    assert "clara_pilot" not in main and "2608.05195" not in main
    shutil.copy2(MANUSCRIPT / "IEEEtran.cls", SOURCE / "IEEEtran.cls")
    shutil.copy2(MANUSCRIPT / "references.tex", SOURCE / "references.tex")
    for figure in sorted((ROOT / "figures_next").glob("*.pdf")):
        shutil.copy2(figure, SOURCE / "figures" / figure.name)
    for tex in ["main.tex", "supplement.tex", "poster_abstract.tex"]:
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", (SOURCE / tex).read_text()):
            assert (SOURCE / "figures" / name).is_file()
    (SOURCE / "README.md").write_text("""# Final CLARA IEEE LaTeX source

Upload this ZIP to Overleaf, select `main.tex` as the main document, and use
pdfLaTeX. The complete main paper and all nineteen references are in this one
file. No BibTeX/Biber step or external input `.tex` file is required. Keep the
included unmodified `IEEEtran.cls` and `figures/` directory beside it.

Local build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The checked main PDF has eight IEEE two-column, letter-size pages including
references. CCED at IEEE BigData 2026 permits eight pages including references
in its workshop CFP. Its generic portal says ten; use the stricter workshop
limit. References are part of the main paper, not an excluded attachment.

The workshop form shows one PDF upload and no separate supplement field.
Upload only the self-contained main PDF. `supplement.tex` and
`poster_abstract.tex` are separate author materials, not extra pages to append
to the submitted PDF. They also contain all of their inputs inline.
`references.tex` is an editable duplicate reference list for convenience;
the main document already embeds it in its `thebibliography` environment.

CCED uses single-blind review, so the author names remain. The acknowledgment
discloses OpenAI ChatGPT (Codex) assistance throughout drafting, mathematics,
software and figure design. Confirm the author details and scientific claims
before submitting. Deadline: 9 October 2026, 11:59 pm Anywhere on Earth.
Notification: 4 November 2026. No external submission has been made.

Official workshop CFP: https://extremedatacurationworkshop.github.io/cfp/
Workshop portal: https://wi-lab.com/cyberchair/2026/bigdata26/scripts/submit.php?subarea=S61

The research/reproducibility ZIP additionally contains frozen data, executed
results, the notebook, tests, proofs and figure-generation software. This ZIP
is a compact LaTeX build package. This revision adds measured serial scaling,
a running example, broader related work, and a replayable typed guard. The
primary registry, thresholds, mathematical certificates, and baseline results
are unchanged.
""")
    shutil.copy2(SOURCE / "main.tex", OUT / "CLARA_IEEE_Final.tex")


def package_sources() -> None:
    archive = OUT / "CLARA_IEEE_LaTeX_Source.zip"
    paths = [SOURCE / name for name in ["README.md", "main.tex", "supplement.tex", "poster_abstract.tex", "references.tex", "IEEEtran.cls"]]
    paths.extend(sorted((SOURCE / "figures").glob("*.pdf")))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(SOURCE).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert bundle.read("main.tex") == (OUT / "CLARA_IEEE_Final.tex").read_bytes()
    print(f"Complete main source with 19 references; {len(paths)} source-package files; {archive.stat().st_size:,} bytes.")


if __name__ == "__main__":
    build_sources()
    package_sources()
