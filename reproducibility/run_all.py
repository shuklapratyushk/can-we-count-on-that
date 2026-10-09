"""Rebuild all frozen-snapshot scientific evidence without network access."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for name in ('test_engine.py','test_completion.py','analyze.py','analyze_completion.py',
             'completion_thresholds.py','analyze_msk.py','benchmark_scale.py',
             'constraint_sensitivity.py','guard_demo.py','test_guard.py','make_scale_tables.py','make_tex_tables.py',
             'make_next_tables.py','make_figures_next.py'):
 print('Running '+name,flush=True)
 command=[sys.executable,str(ROOT/'reproducibility'/name)]
 if name=='guard_demo.py':command.append('--demo')
 subprocess.run(command,cwd=ROOT,check=True)
print('Offline scientific reproduction complete. Compile LaTeX and build_deliverables.py separately.')
