#!/usr/bin/env python
"""Optional: in silico motif insertion with the fine-tuned head (runs when insertion.enabled is true).

Sites: centres of the test evaluation windows, in a fixed random order (seed 0), keeping those with no
peak within insertion.min_distance_to_peak bp and no N within 1 kb, until insertion.n_sites are found.
At each site every motif from insertion.motifs is inserted at the centre in both orientations, plus
insertion.n_shuffles dinucleotide-preserving shuffles of it (re-drawn if equal to the motif or its
reverse complement, or if they contain an insertion.avoid_kmers string). Readout: predicted signal
summed over the site +/- insertion.readout_half_width, per track.

Effect of a motif at a site = mean change of its two orientations minus mean change of its shuffles.
Summary per motif and track: mean and median effect, fraction of sites where the motif beat its
shuffles, and the median log2((ref + motif change) / (ref + shuffle change)), which is scale free so a
track that is simply more sensitive to any insertion does not look motif specific.
These are model hypotheses, not evidence.

Outputs in <run>/insertion/: sites.txt, raw_<start>-<end>.tsv.gz, effects_per_site.tsv.gz, summary.tsv
Usage: 04_motif_insertion.py --config run.yaml [--fold N] [--start i --end j] [--summarize-only] [--smoke]
  --start/--end split the sites over several jobs; then run once with --summarize-only.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agft import common as C  # noqa: E402
from agft.config import EVAL_WINDOW, Config, add_args  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def dinuc_shuffle(seq, rng):
    """Altschul-Erickson dinucleotide shuffle (random Eulerian walk); keeps all dinucleotide counts."""
    if len(seq) < 3:
        return seq
    chars = sorted(set(seq))
    nxt = {c: [] for c in chars}
    for x, y in zip(seq[:-1], seq[1:]):
        nxt[x].append(y)
    last = seq[-1]
    while True:
        last_edge = {c: int(rng.integers(len(nxt[c]))) for c in chars if c != last and nxt[c]}
        ok = True
        for c in last_edge:
            seen, x = set(), c
            while x != last:
                if x in seen:
                    ok = False
                    break
                seen.add(x)
                x = nxt[x][last_edge[x]]
            if not ok:
                break
        if ok:
            break
    order = {}
    for c in chars:
        idx = list(range(len(nxt[c])))
        if c in last_edge:
            e = last_edge[c]
            idx.remove(e)
            rng.shuffle(idx)
            idx.append(e)
        else:
            rng.shuffle(idx)
        order[c] = [nxt[c][i] for i in idx]
    out, ptr, x = [seq[0]], {c: 0 for c in chars}, seq[0]
    for _ in range(len(seq) - 1):
        y = order[x][ptr[x]]
        ptr[x] += 1
        out.append(y)
        x = y
    return "".join(out)


def design(site_i, motifs, n_shuffles, avoid):
    """(motif, kind, sequence) insertions for one site; shuffles are seeded by site and motif."""
    items = []
    for name, m in motifs.items():
        m = m.upper()
        items += [(name, "fwd", m), (name, "rev", C.revcomp(m))]
        rng = np.random.default_rng([site_i, sum(map(ord, name))])
        for k in range(n_shuffles):
            for _ in range(200):  # very short motifs have few distinct shuffles
                sh = dinuc_shuffle(m, rng)
                if sh not in (m, C.revcomp(m)) and not any(b in sh for b in avoid):
                    break
            items.append((name, f"shuffle{k}", sh))
    return items


def pick_sites(cfg, fa, n):
    ins = cfg.insertion
    peaks = C.read_peaks(cfg.peaks)
    test = C.read_windows(cfg.eval_windows, "test")
    d = int(ins["min_distance_to_peak"])
    sites = []
    for w in test.sample(frac=1, random_state=0).itertuples():
        if len(sites) >= n:
            break
        mid = w.start + EVAL_WINDOW // 2
        near = peaks[(peaks.chrom == w.chrom) & (peaks.end > mid - d) & (peaks.start < mid + d)]
        if len(near) or "N" in fa[w.chrom][mid - 1000:mid + 1000]:
            continue
        sites.append(f"{w.chrom}:{mid}")
    return sites


def summarize(cfg, outdir):
    raw = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(outdir.glob("raw_*.tsv.gz"))])
    raw = raw.drop_duplicates(["site", "motif", "kind"])  # overlapping part files count once
    labels = cfg.labels
    dcols = [f"delta_{l}" for l in labels]
    rows = []
    for (site, motif), g in raw[raw.motif != "REF"].groupby(["site", "motif"]):
        mot = g[g.kind.isin(["fwd", "rev"])][dcols].mean()
        sh = g[g.kind.str.startswith("shuffle")][dcols].mean()
        ref = g[[f"ref_{l}" for l in labels]].iloc[0].values
        for j, l in enumerate(labels):
            rows.append(dict(site=site, motif=motif, track=l, ref=ref[j], motif_delta=mot.iloc[j],
                             shuffle_delta=sh.iloc[j], effect=mot.iloc[j] - sh.iloc[j],
                             log2fc=np.log2((ref[j] + mot.iloc[j] + 1e-6) / (ref[j] + sh.iloc[j] + 1e-6))))
    eff = pd.DataFrame(rows)
    eff.to_csv(outdir / "effects_per_site.tsv.gz", sep="\t", index=False)
    summ = eff.groupby(["motif", "track"]).agg(
        n_sites=("site", "nunique"), mean_ref=("ref", "mean"), mean_effect=("effect", "mean"),
        median_effect=("effect", "median"), frac_motif_beats_shuffle=("effect", lambda x: (x > 0).mean()),
        median_log2fc=("log2fc", "median")).reset_index()
    summ.to_csv(outdir / "summary.tsv", sep="\t", index=False)
    print(summ.to_string(index=False))


def main():
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 sites, smoke checkpoint")
    a = ap.parse_args()
    cfg = Config(a.config, a.fold)
    if a.smoke:
        cfg.use_smoke()
    ins = cfg.insertion
    if not ins.get("motifs"):
        sys.exit("insertion.motifs is empty in the config")
    outdir = cfg.run_dir / "insertion"
    outdir.mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        return summarize(cfg, outdir)

    from pyfaidx import Fasta
    from agft import model as M
    fa = Fasta(str(cfg.fasta), as_raw=True, sequence_always_upper=True)
    n = 2 if a.smoke else int(ins["n_sites"])
    sites = pick_sites(cfg, fa, n)
    (outdir / "sites.txt").write_text("\n".join(sites) + "\n")
    end = len(sites) if a.end is None else min(a.end, len(sites))
    print(f"{len(sites)} sites; this job does {a.start} to {end}", flush=True)

    _, head = M.head_specs(cfg)
    model = M.load_finetuned(cfg, M.best_checkpoint(cfg))
    predict = M.make_predict(model, cfg, head)
    CTX, half, nt = cfg.context, int(ins["readout_half_width"]), len(cfg.labels)

    def readout(seq, pos):
        x = M.head_track(predict(C.one_hot(seq)[None]), CTX, nt)
        return x[0, pos - half:pos + half].sum(0)

    rows, t0 = [], time.time()
    for si in range(a.start, end):
        chrom, mid = sites[si].split(":")
        mid = int(mid)
        s, e = C.context_interval(mid, mid, CTX, len(fa[chrom]))
        seq, pos = fa[chrom][s:e], mid - s
        ref = readout(seq, pos)
        rows.append((si, sites[si], "REF", "ref", "", *ref.tolist(), *[0.0] * nt))
        for name, kind, insert in design(si, ins["motifs"], int(ins["n_shuffles"]), ins.get("avoid_kmers") or []):
            i0 = pos - len(insert) // 2
            d = readout(seq[:i0] + insert + seq[i0 + len(insert):], pos) - ref
            rows.append((si, sites[si], name, kind, insert, *ref.tolist(), *d.tolist()))
        if (si - a.start + 1) % 10 == 0:
            print(f"{si - a.start + 1}/{end - a.start} sites  {time.time() - t0:.0f} s", flush=True)
    cols = ["site_idx", "site", "motif", "kind", "inserted"] + [f"ref_{l}" for l in cfg.labels] + \
        [f"delta_{l}" for l in cfg.labels]
    pd.DataFrame(rows, columns=cols).to_csv(outdir / f"raw_{a.start}-{end}.tsv.gz", sep="\t", index=False)
    print(f"done {end - a.start} sites in {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.start == 0 and end == len(sites):
        summarize(cfg, outdir)
    print("INSERTION_DONE")


if __name__ == "__main__":
    main()
