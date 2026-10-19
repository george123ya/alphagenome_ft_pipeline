#!/usr/bin/env python
"""Heads-only fine-tune: the AlphaGenome trunk stays frozen, one new output head is trained.

The head has one output track per bigwig in the config. AdamW, early stopping on validation loss.
Checkpoints (head weights only, small) go to <out>/checkpoints/{best,last}.
The job stops with an error if the head weights did not move (catches the zero-gradient trap,
see agft/model.py head_specs).

With exclude_mode 'mask', bases in exclude_bed loci (widened to whole 128 bp bins) are removed
from the loss: target and prediction are both set to 0 there.

Usage: 02_train.py --config run.yaml [--fold N] [--smoke] [--max-steps N]
  --smoke: 20 steps and 2 validation windows, to test the setup in a few minutes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agft import common as C  # noqa: E402
from agft import model as M  # noqa: E402
from agft.config import Config, add_args  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
from alphagenome.data import genome  # noqa: E402
from alphagenome_ft import create_model_with_heads  # noqa: E402
from alphagenome_ft.finetune import BigWigDataModule, train  # noqa: E402


class MaskedBigWigDataModule(BigWigDataModule):
    """Sets targets to -1 over masked loci (whole 128 bp bins); the loss wrapper zeroes them."""

    def __init__(self, *args, mask=None, **kw):
        super().__init__(*args, **kw)
        self._mask = mask or {}
        self.masked_bp = 0

    def _make_batch(self, batch_indices, windows, extractor, head_handles):
        b = super()._make_batch(batch_indices, windows, extractor, head_handles)
        for j, idx in enumerate(batch_indices):
            w = windows[idx]
            arr = self._mask.get(w.chromosome)
            if arr is None:
                continue
            L = w.end - w.start
            for s, e in arr[(arr[:, 0] < w.end) & (arr[:, 1] > w.start)]:
                s0 = max(0, (int(s) - w.start) // 128 * 128)
                e0 = min(L, -(-(int(e) - w.start) // 128) * 128)
                for k in b:
                    if k.startswith("targets_"):
                        b[k][j, s0:e0, :] = -1.0
                self.masked_bp += e0 - s0
        return b


def install_loss_mask(model):
    orig = model.create_loss_fn_for_head

    def create(name):
        f = orig(name)

        def g(predictions, batch):
            t = batch["targets"]
            m = (t >= 0).astype(t.dtype)
            S = t.shape[1]
            m128 = m.reshape(m.shape[0], S // 128, 128, m.shape[2]).min(axis=2)

            def mk(x):
                if hasattr(x, "ndim") and x.ndim == 3 and x.shape[1] == S:
                    return x * m.astype(x.dtype)
                if hasattr(x, "ndim") and x.ndim == 3 and x.shape[1] == S // 128:
                    return x * m128.astype(x.dtype)
                return x
            return f(jax.tree_util.tree_map(mk, predictions), {**batch, "targets": t * m})
        return g
    model.create_loss_fn_for_head = create


def main():
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    cfg = Config(a.config, a.fold)
    if a.smoke:
        cfg.use_smoke()
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg.train
    t_start = time.time()
    print("devices:", jax.devices(), flush=True)
    if not cfg.train_windows.exists():
        sys.exit(f"{cfg.train_windows} missing: run 01_make_windows.py first")

    specs, head = M.head_specs(cfg)
    win = C.read_windows(cfg.train_windows)
    tr, va = win[win.split == "train"], win[win.split == "valid"]
    nvalid = 2 if a.smoke else tc.get("max_valid_windows")
    if nvalid:
        va = va.head(int(nvalid))
    to_iv = lambda d: [genome.Interval(r.chrom, int(r.start), int(r.end)) for r in d.itertuples()]  # noqa: E731
    intervals = {"train": to_iv(tr), "valid": to_iv(va)}
    print({k: len(v) for k, v in intervals.items()}, flush=True)
    mask = C.read_bed_intervals(cfg.exclude_bed) if cfg.exclude_bed and cfg.exclude_mode == "mask" else None
    dm = MaskedBigWigDataModule(intervals=intervals, fasta_path=Path(cfg.fasta), head_specs=specs,
                                batch_size=int(tc["batch_size"]), shuffle=True, drop_last=True, mask=mask)

    t0 = time.time()
    model = create_model_with_heads("all_folds", heads=[head], checkpoint_path=str(cfg.weights),
                                    organism_settings=M.organism_settings(cfg), device=jax.devices()[0],
                                    detach_backbone=True)
    info = {"config": cfg.summary(), "smoke": a.smoke, "model_load_s": round(time.time() - t0)}
    print(f"model loaded in {info['model_load_s']} s", flush=True)
    if mask:
        install_loss_mask(model)

    before = {k: {kk: np.asarray(vv).copy() for kk, vv in v.items()}
              for k, v in model._params.items() if k.startswith("head/")}
    max_steps = 20 if a.smoke else a.max_steps
    t0 = time.time()
    train(model=model, data_module=dm, head_specs=specs, learning_rate=float(tc["learning_rate"]),
          weight_decay=float(tc["weight_decay"]), num_epochs=int(tc["epochs"]), seed=int(tc["seed"]),
          max_train_steps=max_steps, heads_only=True, checkpoint_dir=cfg.ckpt_dir, organism=cfg.organism,
          best_metric="valid_loss", best_metric_mode="min",
          early_stopping_patience=int(tc["early_stopping_patience"]), verbose=True)
    info["train_minutes"] = round((time.time() - t0) / 60, 1)
    info["masked_bp_seen"] = dm.masked_bp
    info["peak_gpu_gib"] = M.gpu_memory_gib()
    change = max(float(np.abs(np.asarray(v) - before[k][kk]).max())
                 for k, sub in model._params.items() if k.startswith("head/") for kk, v in sub.items())
    info["head_param_max_change"] = change
    info["total_minutes"] = round((time.time() - t_start) / 60, 1)
    json.dump(info, open(cfg.run_dir / "train_info.json", "w"), indent=1)
    print(f"training {info['train_minutes']} min; head weights max change {change:.4g}", flush=True)
    if change < 1e-4:
        sys.exit("ERROR: head weights did not change, training had no effect")
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
