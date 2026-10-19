#!/usr/bin/env python
"""Check a run config and its input files before using the GPU. CPU only, about a minute.

Stops with a clear message if: a file is missing, the genome is not the assembly AlphaGenome
was trained on (hg38 / mm10), bigwig chromosome names or lengths do not match the FASTA, a split
chromosome is missing, peaks use other chromosome names, or the weights folder looks wrong.
Prints warnings for things that work but are worth a look. Writes <out>/checks.txt.

Usage: 00_check_inputs.py --config run.yaml [--fold N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agft import common as C  # noqa: E402
from agft.config import ASSEMBLY_CHR1, Config, ConfigError, add_args  # noqa: E402

errors, warnings, notes = [], [], []


def main():
    a = add_args(argparse.ArgumentParser()).parse_args()
    try:
        cfg = Config(a.config, a.fold)
    except (ConfigError, FileNotFoundError, KeyError) as e:
        sys.exit(f"CONFIG ERROR: {e}")

    import numpy as np
    import pyBigWig
    from pyfaidx import Fasta

    files = [cfg.fasta, *cfg.bigwigs, *cfg.peaks] + ([cfg.exclude_bed] if cfg.exclude_bed else [])
    for f in files:
        if not Path(f).exists():
            errors.append(f"missing file: {f}")
    if not (cfg.weights / "_METADATA").exists():
        errors.append(f"weights folder {cfg.weights} has no _METADATA: download google/alphagenome-all-folds there")
    if errors:
        return finish(cfg)

    fa = Fasta(str(cfg.fasta), as_raw=True)
    lens = {k: len(fa[k]) for k in fa.keys()}
    asm, chr1 = ASSEMBLY_CHR1[cfg.organism]
    if lens.get("chr1") != chr1:
        errors.append(f"FASTA chr1 is {lens.get('chr1')} bp; {cfg.organism} in AlphaGenome is {asm} "
                      f"(chr1 {chr1} bp). Use {asm} (or lift your data over to it).")
    chroms = cfg.chromosomes or C.default_chromosomes(lens)
    for c in cfg.test + cfg.valid:
        if c not in lens:
            errors.append(f"split chromosome {c} is not in the FASTA")
        if c not in chroms:
            errors.append(f"split chromosome {c} is not in the chromosome list used for windows")
    train = [c for c in chroms if c not in cfg.test + cfg.valid]
    notes.append(f"chromosomes: test {cfg.test}, valid {cfg.valid}, train {train}")

    for lab, bwp in zip(cfg.labels, cfg.bigwigs):
        bw = pyBigWig.open(str(bwp))
        bl = bw.chroms()
        shared = [c for c in chroms if c in bl]
        if not shared:
            errors.append(f"{lab}: no chromosome names in common with the FASTA (chr1 vs 1?)")
            continue
        bad = [c for c in shared if bl[c] != lens[c]]
        if bad:
            errors.append(f"{lab}: chromosome lengths differ from the FASTA for {bad[:5]} (other assembly?)")
        missing = [c for c in chroms if c not in bl]
        if missing:
            warnings.append(f"{lab}: no signal for {missing}; windows there train on zeros")
        # coarse signal summary on chr1-sized sample
        c0 = shared[0]
        v = np.nan_to_num(np.asarray(bw.values(c0, 0, min(bl[c0], 20_000_000)), np.float32))
        if (v < 0).any():
            errors.append(f"{lab}: negative values in the bigwig; coverage-type heads need values >= 0")
        frac = float((v > 0).mean())
        notes.append(f"{lab}: {Path(bwp).name}, {c0}:0-20Mb mean {v.mean():.4g}, max {v.max():.4g}, "
                     f"nonzero {frac:.1%}")
        if v.max() == 0:
            errors.append(f"{lab}: all zero on {c0}:0-20Mb")
        bw.close()

    peaks = C.read_peaks(cfg.peaks)
    if peaks.empty:
        errors.append("no peaks read")
    else:
        bad = sorted(set(peaks.chrom) - set(lens))
        in_use = peaks[peaks.chrom.isin(chroms)]
        notes.append(f"peaks: {len(peaks):,} in {len(cfg.peaks)} files; {len(in_use):,} on used chromosomes; "
                     f"median width {int((peaks.end - peaks.start).median())} bp")
        if len(in_use) == 0:
            errors.append("no peaks on the used chromosomes (naming mismatch?)")
        elif bad:
            warnings.append(f"peaks on chromosomes not in the FASTA (ignored): {bad[:5]}")
        for c in cfg.test:
            if (peaks.chrom == c).sum() == 0:
                warnings.append(f"no peaks on test chromosome {c}")

    if cfg.exclude_bed:
        ex = C.read_bed_intervals(cfg.exclude_bed)
        notes.append(f"exclude: {sum(len(v) for v in ex.values()):,} loci, mode {cfg.exclude_mode}")
        if cfg.exclude_mode == "drop" and cfg.context > 131072:
            warnings.append("exclude_mode drop at long context can remove many windows; 'mask' keeps them")

    if cfg.kind in ("chip_tf", "chip_histone"):
        notes.append(f"{cfg.kind} heads predict at 128 bp only; evaluation spreads each bin over 128 bp")
    if cfg.kind == "rna_seq":
        warnings.append("rna_seq: strand-specific tracks need one bigwig per strand; check the head docs")
    if cfg.context == 1048576:
        notes.append("1 Mb context needs about 28 GB of GPU memory (A100 or L40S, not a 16 GB card)")
    return finish(cfg)


def finish(cfg):
    cfg.out.mkdir(parents=True, exist_ok=True)
    lines = [f"config {cfg.path}", json.dumps(cfg.summary(), indent=1)]
    lines += [f"NOTE  {x}" for x in notes] + [f"WARN  {x}" for x in warnings] + [f"ERROR {x}" for x in errors]
    lines.append("RESULT " + ("FAILED" if errors else "OK"))
    (cfg.out / "checks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:]))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
