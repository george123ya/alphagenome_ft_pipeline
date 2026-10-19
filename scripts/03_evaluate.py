#!/usr/bin/env python
"""Score the fine-tuned head on the held-out test chromosomes.

Uses the best checkpoint (lowest validation loss). For each 131,072 bp evaluation window the model
reads `context` bp centred on it and only the central 131,072 bp are scored, so runs with different
contexts are compared on exactly the same bases.

Outputs in the run folder:
- test_metrics.tsv: per track, Pearson (log1p) and Spearman of signal summed per peak (union of peak
  summits +/- evaluate.peak_half_width), Pearson (log1p) at 128 bp bins; per compare pair, Pearson
  of the between-track log ratio at the same peaks (can the model tell the samples apart?).
- test_peak_sums.tsv.gz: observed and predicted sum per peak.
- test_bins128.npz: observed and predicted 128 bp bins.
- eval_info.json: timing and counts.

Usage: 03_evaluate.py --config run.yaml [--fold N] [--smoke] [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agft import common as C  # noqa: E402
from agft import model as M  # noqa: E402
from agft.config import EVAL_WINDOW, Config, add_args  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyBigWig  # noqa: E402
from pyfaidx import Fasta  # noqa: E402


def main():
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--smoke", action="store_true", help="3 windows, smoke checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test windows")
    a = ap.parse_args()
    cfg = Config(a.config, a.fold)
    if a.smoke:
        cfg.use_smoke()
    T0 = time.time()
    _, head = M.head_specs(cfg)
    ck = M.best_checkpoint(cfg)
    model = M.load_finetuned(cfg, ck)
    predict = M.make_predict(model, cfg, head)
    nt, CTX, W = len(cfg.labels), cfg.context, EVAL_WINDOW

    fa = Fasta(str(cfg.fasta), as_raw=True, sequence_always_upper=True)
    bws = [pyBigWig.open(str(p)) for p in cfg.bigwigs]
    peaks = C.union_summits(C.read_peaks(cfg.peaks), int(cfg.evaluate["peak_half_width"]))
    test = C.read_windows(cfg.eval_windows, "test")
    limit = 3 if a.smoke else a.limit
    if limit:
        test = test.head(limit)
    print(f"checkpoint {ck}; {len(test)} test windows; context {CTX}", flush=True)

    rows, bin_o, bin_p, t_first = [], [], [], None
    for i, w in enumerate(test.itertuples()):
        s, e = C.context_interval(w.start, w.end, CTX, len(fa[w.chrom]))
        pred = M.head_track(predict(C.one_hot(fa[w.chrom][s:e])[None]), CTX, nt)[0]
        p1 = pred[w.start - s:w.start - s + W]
        if t_first is None:
            t_first = time.time()
        obs = np.stack([np.nan_to_num(np.asarray(bw.values(w.chrom, int(w.start), int(w.end)), np.float32))
                        if w.chrom in bw.chroms() else np.zeros(W, np.float32) for bw in bws], -1)
        for ro, rp in zip(C.peak_sums(w.chrom, w.start, w.end, obs, peaks),
                          C.peak_sums(w.chrom, w.start, w.end, p1, peaks)):
            rows.append(ro + rp[3:])
        bin_o.append(obs.reshape(-1, 128, nt).sum(1))
        bin_p.append(p1.reshape(-1, 128, nt).sum(1))
        if i % 100 == 0:
            print(f"window {i}/{len(test)}  {time.time() - T0:.0f} s", flush=True)

    oc, pc = [f"obs_{l}" for l in cfg.labels], [f"pred_{l}" for l in cfg.labels]
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"] + oc + pc)
    df.to_csv(cfg.run_dir / "test_peak_sums.tsv.gz", sep="\t", index=False)
    res = C.correlations(df, cfg.labels, cfg.compare_pairs())
    bo, bp = np.log1p(np.concatenate(bin_o).clip(0)), np.log1p(np.concatenate(bin_p).clip(0))
    for j, l in enumerate(cfg.labels):
        res.loc[len(res)] = dict(metric="bin128_log1p_pearson", track=l, value=np.corrcoef(bo[:, j], bp[:, j])[0, 1])
    res.to_csv(cfg.run_dir / "test_metrics.tsv", sep="\t", index=False)
    np.savez_compressed(cfg.run_dir / "test_bins128.npz", obs=np.concatenate(bin_o).astype(np.float32),
                        pred=np.concatenate(bin_p).astype(np.float32), labels=np.array(cfg.labels),
                        windows=test[["chrom", "start", "end"]].astype(str).values)
    info = dict(checkpoint=str(ck), n_test_windows=len(test), n_test_peaks=len(df),
                seconds_per_window=round((time.time() - t_first) / max(1, len(test) - 1), 3),
                peak_gpu_gib=M.gpu_memory_gib(), total_minutes=round((time.time() - T0) / 60, 1))
    json.dump(info, open(cfg.run_dir / "eval_info.json", "w"), indent=1)
    print(res.to_string(index=False))
    print("EVAL_DONE", info)


if __name__ == "__main__":
    main()
