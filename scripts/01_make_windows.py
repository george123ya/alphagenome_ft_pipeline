#!/usr/bin/env python
"""Cut the genome into windows for training and evaluation. CPU only, a few minutes.

Writes, inside the run folder:
- windows/windows_<context>.bed: train and valid windows (also test ones, unused for training).
  Tiles of `context` bp from position 0 with no overlap. For contexts above 131,072 bp an extra
  tile ending at the chromosome end is added so the tail is not lost.
- windows/eval_windows_131072.bed: 131,072 bp tiles on the test chromosomes. Evaluation always
  scores these, whatever the context, so runs with different contexts are comparable.
Columns: chrom, start, end, split, n_peaks, n_exclude_loci. A .stats.txt file sits next to each.

Rules: keep windows with at least windows.min_peaks peaks (any peak file) and an N fraction below
windows.max_n_fraction. Windows touching exclude_bed loci are dropped when exclude_mode is 'drop';
with 'mask' they are kept and the loss ignores those loci (see 02_train.py). Evaluation windows
always drop them.

Usage: 01_make_windows.py --config run.yaml [--fold N]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agft import common as C  # noqa: E402
from agft.config import EVAL_WINDOW, Config, add_args  # noqa: E402

from pyfaidx import Fasta  # noqa: E402


def tiles(L, W, tail):
    starts = list(range(0, L - W + 1, W))
    if tail and starts and starts[-1] + W < L:
        starts.append(L - W)
    return starts


def make(cfg, fa, chroms, W, peaks, excl, only_test, drop_excluded, tail, out):
    stats, rows = defaultdict(int), []
    for c in chroms:
        split = "test" if c in cfg.test else "valid" if c in cfg.valid else "train"
        if only_test and split != "test":
            continue
        for s in tiles(len(fa[c]), W, tail):
            e = s + W
            stats["tiles"] += 1
            npk = C.n_overlap(peaks.get(c), s, e)
            if npk < cfg.win["min_peaks"]:
                stats["dropped_few_peaks"] += 1
                continue
            nex = C.n_overlap(excl.get(c), s, e)
            if nex and drop_excluded:
                stats["dropped_exclude_bed"] += 1
                continue
            if fa[c][s:e].count("N") / W >= cfg.win["max_n_fraction"]:
                stats["dropped_too_many_N"] += 1
                continue
            stats[f"kept_{split}"] += 1
            if nex:
                stats[f"kept_{split}_with_masked_loci"] += 1
            rows.append((c, s, e, split, npk, nex))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    with open(out.with_suffix(".stats.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}\t{v}\n")
    print(out.name, dict(stats), flush=True)
    return stats


def main():
    a = add_args(argparse.ArgumentParser()).parse_args()
    cfg = Config(a.config, a.fold)
    fa = Fasta(str(cfg.fasta), as_raw=True, sequence_always_upper=True)
    chroms = cfg.chromosomes or C.default_chromosomes(list(fa.keys()))
    p = C.read_peaks(cfg.peaks)
    peaks = {c: g[["start", "end"]].values for c, g in p.groupby("chrom")}
    excl = C.read_bed_intervals(cfg.exclude_bed) if cfg.exclude_bed else {}

    st = make(cfg, fa, chroms, cfg.context, peaks, excl, only_test=False,
              drop_excluded=cfg.exclude_mode == "drop", tail=cfg.context > EVAL_WINDOW, out=cfg.train_windows)
    make(cfg, fa, chroms, EVAL_WINDOW, peaks, excl, only_test=True, drop_excluded=True, tail=False,
         out=cfg.eval_windows)
    for split in ("train", "valid"):
        if not st.get(f"kept_{split}"):
            sys.exit(f"no {split} windows kept: check peaks, split and min_peaks")


if __name__ == "__main__":
    main()
