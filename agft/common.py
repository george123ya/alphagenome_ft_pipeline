"""Data helpers that do not need JAX: BED/peak reading, windows, one-hot, evaluation metrics."""
import gzip
from collections import defaultdict

import numpy as np
import pandas as pd


def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _bed_rows(path):
    with _open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            yield line.rstrip("\n").split("\t")


def read_bed_intervals(path):
    """{chrom: sorted (n, 2) int array of [start, end)} from any BED-like file."""
    d = defaultdict(list)
    for p in _bed_rows(path):
        d[p[0]].append((int(p[1]), int(p[2])))
    return {c: np.array(sorted(v), dtype=np.int64) for c, v in d.items()}


def read_peaks(paths):
    """All peak files stacked. Summit = narrowPeak column 10 when present, else the midpoint."""
    rows = []
    for path in paths:
        for p in _bed_rows(path):
            s, e = int(p[1]), int(p[2])
            summit = s + int(p[9]) if len(p) >= 10 and p[9] not in ("", ".") and int(p[9]) >= 0 else (s + e) // 2
            rows.append((p[0], s, e, summit, str(path)))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "summit_pos", "source"])


def union_summits(peaks, half):
    """Union of summits from all peak files, +/- half bp, merged greedily so pieces never overlap."""
    out = []
    for c, g in peaks.groupby("chrom"):
        last = -10**12
        for p in np.sort(g.summit_pos.values):
            if p - last > 2 * half:
                out.append((c, p - half, p + half))
                last = p
    return pd.DataFrame(out, columns=["chrom", "start", "end"])


def n_overlap(arr, s, e):
    if arr is None or len(arr) == 0:
        return 0
    return int(((arr[:, 0] < e) & (arr[:, 1] > s)).sum())


def read_windows(path, split=None):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "split"])
    return df if split is None else df[df.split == split].reset_index(drop=True)


def default_chromosomes(fasta_keys):
    """chr1..chrN plus chrX: no chrY, chrM, random, unplaced or alt contigs."""
    keep = [c for c in fasta_keys if c.startswith("chr") and c[3:].isdigit()]
    keep = sorted(keep, key=lambda c: int(c[3:]))
    return keep + (["chrX"] if "chrX" in fasta_keys else [])


def one_hot(seq):
    lut = np.zeros((256, 4), np.float32)
    for i, b in enumerate("ACGT"):
        lut[ord(b)] = lut[ord(b.lower())] = np.eye(4, dtype=np.float32)[i]
    return lut[np.frombuffer(seq.encode(), np.uint8)]


def revcomp(s):
    return s[::-1].translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def context_interval(start, end, context, chrom_len):
    """A `context` bp interval centred on [start, end), shifted to stay inside the chromosome."""
    mid = (start + end) // 2
    s = max(0, min(mid - context // 2, chrom_len - context))
    return s, s + context


def peak_sums(chrom, start, end, track, peaks):
    """Sum a (L, T) 1 bp track over the peaks that lie fully inside [start, end)."""
    sub = peaks[(peaks.chrom == chrom) & (peaks.start >= start) & (peaks.end <= end)]
    return [(chrom, r.start, r.end, *track[r.start - start:r.end - start].sum(0).tolist())
            for r in sub.itertuples()]


def correlations(df, labels, pairs):
    """Per track: Pearson on log1p and Spearman of per-peak sums. Per pair: Pearson of log ratios."""
    oc, pc = [f"obs_{l}" for l in labels], [f"pred_{l}" for l in labels]
    lo, lp = np.log1p(df[oc].clip(lower=0)), np.log1p(df[pc].clip(lower=0))
    res = []
    for l, o, p in zip(labels, oc, pc):
        res.append(dict(metric="peak_log1p_pearson", track=l, value=np.corrcoef(lo[o], lp[p])[0, 1]))
        res.append(dict(metric="peak_spearman", track=l, value=df[o].rank().corr(df[p].rank())))
    for a, b in pairs:
        ro = lo[f"obs_{a}"] - lo[f"obs_{b}"]
        rp = lp[f"pred_{a}"] - lp[f"pred_{b}"]
        res.append(dict(metric="peak_logratio_pearson", track=f"{a}_vs_{b}", value=np.corrcoef(ro, rp)[0, 1]))
    return pd.DataFrame(res)
