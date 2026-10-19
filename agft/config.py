"""Read and check the run config (one YAML file per experiment).

Every script takes `--config FILE` and, optionally, `--fold N`. All paths and settings come from
the config, so a new dataset needs a new YAML file and no code changes. See config/template.yaml.
"""
import json
import re
from pathlib import Path

import yaml

SUPPORTED_CONTEXTS = (131072, 524288, 1048576)
SUPPORTED_KINDS = ("atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone")
# AlphaGenome was trained on hg38 (human) and mm10 (mouse). chr1 length identifies the assembly.
ASSEMBLY_CHR1 = {"HOMO_SAPIENS": ("hg38", 248956422), "MUS_MUSCULUS": ("mm10", 195471971)}
EVAL_WINDOW = 131072  # test windows are always scored in 131,072 bp pieces

DEFAULTS = {
    "exclude_bed": None,
    "exclude_mode": "drop",
    "chromosomes": None,
    "windows": {"min_peaks": 1, "max_n_fraction": 0.05},
    "train": {"epochs": 3, "batch_size": 1, "learning_rate": 1e-3, "weight_decay": 1e-4,
              "early_stopping_patience": 1, "max_valid_windows": None, "seed": 42},
    "evaluate": {"peak_half_width": 250, "compare_pairs": "all"},
    "insertion": {"enabled": False, "n_sites": 300, "n_shuffles": 5, "readout_half_width": 250,
                  "min_distance_to_peak": 5000, "avoid_kmers": [], "motifs": {}},
}


class ConfigError(ValueError):
    pass


def _merge(base, extra):
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


class Config:
    """Parsed config. `fold` fills `{fold}` in split.folds_json and adds a fold_N output folder."""

    def __init__(self, path, fold=None):
        self.path = Path(path).resolve()
        raw = yaml.safe_load(open(self.path))
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: not a YAML mapping")
        self.raw = _merge(DEFAULTS, raw)
        self.base_dir = self.path.parent
        r = self.raw
        for key in ("name", "out_dir", "organism", "genome_fasta", "weights", "context", "head", "peaks", "split"):
            if key not in r or r[key] in (None, "", []):
                raise ConfigError(f"config is missing '{key}'")
        self.name = str(r["name"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ConfigError("name may only use letters, digits, _ . - (it is used in job names)")
        self.organism = str(r["organism"]).upper()
        if self.organism not in ASSEMBLY_CHR1:
            raise ConfigError(f"organism must be HOMO_SAPIENS or MUS_MUSCULUS, got {self.organism}")
        self.fasta = self._path(r["genome_fasta"])
        self.weights = self._path(r["weights"])
        self.context = int(r["context"])
        if self.context not in SUPPORTED_CONTEXTS:
            raise ConfigError(f"context must be one of {SUPPORTED_CONTEXTS}, got {self.context}")

        h = r["head"]
        self.kind = str(h.get("kind", "")).lower()
        if self.kind not in SUPPORTED_KINDS:
            raise ConfigError(f"head.kind must be one of {SUPPORTED_KINDS}, got '{self.kind}'")
        self.head_id = str(h.get("id") or f"{self.name}_{self.kind}")
        tracks = h.get("tracks") or []
        if not tracks:
            raise ConfigError("head.tracks is empty")
        self.labels = [str(t["label"]) for t in tracks]
        self.bigwigs = [self._path(t["bigwig"]) for t in tracks]
        if len(set(self.labels)) != len(self.labels):
            raise ConfigError("head.tracks labels must be unique")
        self.head_overrides = {k: h[k] for k in ("resolutions", "apply_squashing") if k in h}

        self.peaks = [self._path(p) for p in r["peaks"]]
        self.exclude_bed = self._path(r["exclude_bed"]) if r["exclude_bed"] else None
        self.exclude_mode = str(r["exclude_mode"]).lower()
        if self.exclude_mode not in ("drop", "mask"):
            raise ConfigError("exclude_mode must be 'drop' or 'mask'")

        self.fold = fold
        s = r["split"]
        if "folds_json" in s:
            if "{fold}" in str(s["folds_json"]) and fold is None:
                raise ConfigError("split.folds_json contains {fold}: pass --fold N")
            fj = self._path(str(s["folds_json"]).replace("{fold}", str(fold)))
            d = json.load(open(fj))
            self.test, self.valid = list(d["test"]), list(d["valid"])
            self.split_source = str(fj)
        else:
            self.test, self.valid = list(s.get("test") or []), list(s.get("valid") or [])
            self.split_source = "inline"
        if not self.test or not self.valid:
            raise ConfigError("split needs test and valid chromosomes (or folds_json)")
        if set(self.test) & set(self.valid):
            raise ConfigError(f"chromosomes in both test and valid: {set(self.test) & set(self.valid)}")
        self.chromosomes = r["chromosomes"]

        self.win = r["windows"]
        self.train = r["train"]
        self.evaluate = r["evaluate"]
        self.insertion = r["insertion"]

        out = Path(r["out_dir"]).expanduser()
        out = out if out.is_absolute() else self.base_dir / out
        self.out = out / f"fold_{fold}" if fold is not None else out
        self.run_dir = self.out  # checkpoints and results; windows always stay in out/windows

    def use_smoke(self):
        """Smoke tests write checkpoints and results to out/smoke, never over a real run."""
        self.run_dir = self.out / "smoke"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, p):
        p = Path(str(p)).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    # ---- standard file locations inside the run folder ----
    @property
    def train_windows(self):
        return self.out / "windows" / f"windows_{self.context}.bed"

    @property
    def eval_windows(self):
        return self.out / "windows" / f"eval_windows_{EVAL_WINDOW}.bed"

    @property
    def ckpt_dir(self):
        return self.run_dir / "checkpoints"

    def compare_pairs(self):
        """Pairs of track labels whose log ratio is compared (first minus second)."""
        cp = self.evaluate.get("compare_pairs", "all")
        if cp in (None, "none", []):
            return []
        if cp == "all":
            return [(self.labels[i], self.labels[j]) for i in range(len(self.labels)) for j in range(i)]
        pairs = [tuple(p) for p in cp]
        for a, b in pairs:
            if a not in self.labels or b not in self.labels:
                raise ConfigError(f"evaluate.compare_pairs: unknown label in {a, b}")
        return pairs

    def targets_config(self):
        """The dict alphagenome_ft.finetune.load_targets_config expects."""
        head = {"id": self.head_id, "source": "predefined", "kind": self.kind,
                "targets": [{"path": str(p), "label": l} for p, l in zip(self.bigwigs, self.labels)]}
        head.update(self.head_overrides)
        return {"heads": [head]}

    def summary(self):
        return {"name": self.name, "config": str(self.path), "fold": self.fold, "out": str(self.out),
                "organism": self.organism, "context": self.context, "kind": self.kind,
                "tracks": dict(zip(self.labels, map(str, self.bigwigs))), "test": self.test,
                "valid": self.valid, "split_source": self.split_source,
                "exclude": str(self.exclude_bed) if self.exclude_bed else None,
                "exclude_mode": self.exclude_mode, "train": self.train}


def add_args(ap):
    ap.add_argument("--config", required=True, help="run config YAML")
    ap.add_argument("--fold", type=int, default=None, help="fold number, fills {fold} in split.folds_json")
    return ap
