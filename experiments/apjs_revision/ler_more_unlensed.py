"""Top up the detectable unlensed draw of a configuration (same LeR settings, new seed) and merge it
into n_unlensed_detectable_<config>.json; the original file is kept as *_first_draw.json."""
import argparse, json, shutil
from pathlib import Path
import numpy as np
from ler_population_rev import build_ler

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--work-dir", required=True)
ap.add_argument("--extra", type=int, default=2000); ap.add_argument("--seed", type=int, default=20261107 + 5)
ap.add_argument("--npool", type=int, default=16)
a = ap.parse_args()
work = Path(a.work_dir); d = work / f"ler_data_{a.config}"
np.random.seed(a.seed)
ler = build_ler(a.config, work, a.npool, 8.0, 500.0)
up = ler.selecting_n_unlensed_detectable_events(size=a.extra, batch_size=50000, pdet_threshold=0.5, resume=False,
                                                output_jsonfile=f"n_unlensed_detectable_{a.config}_extra.json",
                                                meta_data_file=f"meta_unlensed_{a.config}_extra.json", trim_to_size=True)
up = up[1] if isinstance(up, tuple) else up
main = d / f"n_unlensed_detectable_{a.config}.json"; first = d / f"n_unlensed_detectable_{a.config}_first_draw.json"
if not first.exists():
    shutil.copy(main, first)
base = json.loads(first.read_text()); extra = json.loads((d / f"n_unlensed_detectable_{a.config}_extra.json").read_text())
merged = {k: list(base[k]) + list(extra[k]) for k in base if k in extra}
main.write_text(json.dumps(merged))
print({k: len(v) for k, v in list(merged.items())[:3]}, "merged; seed", a.seed)
