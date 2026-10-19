# AlphaGenome heads-only fine-tuning pipeline

Train a small new output head on top of the frozen AlphaGenome model (Google DeepMind, 2025) so it
predicts your own genomic tracks (ATAC, DNase, ChIP, CUT&RUN, CAGE, RNA-seq) from DNA sequence.
Then score it on held-out chromosomes and, optionally, insert motifs in silico to ask which sequences
the model thinks open chromatin in which sample.

Everything about one experiment lives in a single YAML config. A new dataset needs a new config, not
code changes. The worked example is the Fuchs lab SOX2/SOX9 cell line ATAC data
(`config/example_sox_cell_lines_atac.yaml`).

Locations:
- RU HPC: `/lustre/fs4/fuchs_lab/scratch/gmunoz/alphagenome_ft_pipeline`
- genos: `/home/genos/gmunoz/SCRATCH/Projects/alphagenome_ft_pipeline`

## What the pipeline does

| Step | Script | Where | Time (131 kb context) |
|---|---|---|---|
| 00 | `scripts/00_check_inputs.py`: checks files, assembly, chromosome names, signal | CPU | 1 min |
| 01 | `scripts/01_make_windows.py`: cuts the genome into windows, train/valid/test by chromosome | CPU | 1 min |
| 02 | `scripts/02_train.py`: heads-only fine-tune, early stopping, checkpoints | GPU | about 30 min |
| 03 | `scripts/03_evaluate.py`: accuracy on the test chromosomes | GPU | about 5 min |
| 04 | `scripts/04_motif_insertion.py`: optional motif insertion | GPU | about 30 min |

`run_pipeline.sh` runs all steps for one config and one fold, and skips steps that already finished,
so a crashed job can just be submitted again. `submit.sh` sends one SLURM job per fold.

The AlphaGenome trunk is frozen (`detach_backbone=True`, no gradient into it), which is why this fits
on one GPU. Only the new head learns. AdamW, learning rate 1e-3, batch 1, up to 3 epochs, early
stopping on validation loss.

## Quick start (RU HPC)

1. **Weights licence (once, personal).** Accept the AlphaGenome terms at
   https://huggingface.co/google/alphagenome-all-folds (non-commercial use only; it also covers
   fine-tuned models and their outputs). Then either download your own copy with
   `bash env/download_weights.sh /lustre/.../YOUR_USER/alphagenome_weights/all_folds`, or, once you
   have accepted, ask George about using the lab copy listed in the example config.
2. **Environment.** You can use the tested environment through `env.sh` (the default `AG_ENV`) if you
   can read it, or build your own (about 10 min, about 6 GB):

       sbatch -p hpc_l40_a -c 4 --mem=16G -t 01:30:00 --wrap "bash env/install_env.sh /lustre/.../YOUR_USER/envs/alphagenome"
       export AG_ENV=/lustre/.../YOUR_USER/envs/alphagenome    # put this in ~/.bashrc

3. **Config.** `cp config/template.yaml config/my_experiment.yaml` and fill in the REQUIRED part:
   genome FASTA, weights, bigwigs (one per sample), peak files, and the chromosome split.
4. **Check the inputs** (CPU, 1 to 2 min):

       srun -p hpc_l40_a -c 4 --mem=16G -t 30 env STEPS="00 01" bash run_pipeline.sh config/my_experiment.yaml 0

   It stops with a clear message on the usual problems (wrong assembly, `chr1` vs `1`, missing files).
   Read `<out_dir>/fold_0/checks.txt` and `windows/*.stats.txt`: you want thousands of train windows.
5. **Smoke test** (about 10 to 15 min on a GPU; writes to `<out_dir>/fold_0/smoke`, never over a real run):

       SMOKE=1 GPU=l40 bash submit.sh config/my_experiment.yaml 0

   The log must end with `PIPELINE_DONE`, and step 02 must print a head weight change well above 1e-4.
6. **Real runs, all folds in parallel:**

       GPU=l40 bash submit.sh config/my_experiment.yaml 0 1 2 3 4
       squeue -u $USER
       tail -f <out_dir>/logs/*.log

`GPU=a100` (default) uses the single A100 node (80 GB, often busy); `GPU=l40` uses the L40 nodes
(48 GB, 10 nodes, usually free); `GPU=l40s` the L40S nodes. All of them fit the 1 Mb context.

## Inputs and their rules

- **Assembly.** AlphaGenome was trained on **mm10** (mouse) and **hg38** (human). Your FASTA, bigwigs
  and peaks must be on that assembly. Step 00 refuses mm39, hg19 and Ensembl-style names (`1`).
- **Bigwigs.** One per output track, values >= 0. Coverage (CPM or raw counts) works; pool replicates
  into one bigwig per sample. Missing chromosomes count as zero signal.
- **Peaks.** narrowPeak or BED, any number of files. They decide which windows are used (a window needs
  `windows.min_peaks` peaks) and which positions are scored in evaluation (union of summits +/- 250 bp).
- **Split.** By whole chromosomes, so nothing in test is seen in training. `folds/mm10/fold_0..4.json`
  are the five mouse folds used by the lab's ChromBPNet models. For human, write your own fold files
  or put `test:` and `valid:` lists in the config.
- **exclude_bed** (optional): blacklist or contamination loci. `drop` removes windows that touch them
  (fine at 131 kb). `mask` keeps the windows and removes only those bases from the loss (better at 1 Mb,
  where almost every window touches something). Evaluation always drops them.
- **Context.** 131,072 bp is fast and fits a 16 GB GPU. 1,048,576 bp (1 Mb) needs about 28 GB and is
  about 3 times slower per run; in the SOX example it improved the between-sample comparison a little
  (log-ratio r 0.43 to 0.47). Start with 131 kb.
- **Head kind.** `atac`, `dnase`, `procap`, `cage`, `rna_seq`, `chip_tf`, `chip_histone`. ATAC and
  CUT&RUN-like coverage: use `atac` (or `chip_tf` for TF ChIP; it predicts at 128 bp only).

## Outputs (in `<out_dir>/fold_N/`)

- `checks.txt`, `windows/` (with `.stats.txt` counts)
- `checkpoints/best`, `checkpoints/last`: the trained head (small; the trunk is not copied)
- `train_info.json`: minutes, GPU memory, how much the head weights moved
- `test_metrics.tsv`, the main result:
  - `peak_log1p_pearson`, `peak_spearman`: per sample, predicted vs observed signal per peak
  - `peak_logratio_pearson`: for each pair in `evaluate.compare_pairs`, does the model predict the
    *difference* between two samples at the same peaks? This is usually the interesting number and
    is much lower than the per-sample one.
  - `bin128_log1p_pearson`: at 128 bp resolution, over whole test windows
- `test_peak_sums.tsv.gz`, `test_bins128.npz`: the numbers behind the metrics, for plots
- `insertion/summary.tsv` (if enabled): per motif and sample, effect versus dinucleotide shuffles

For reference, SOX cell lines fold 0 at 131 kb: per-sample peak Pearson 0.68 to 0.77; Sox2hi vs
Sox9hi log ratio 0.43; Double vs Sox9hi log ratio about 0.1 (not predictable). The best pretrained
AlphaGenome track without fine-tuning reached only 0.50 to 0.61 per sample.

## Run times measured

Measured on the SOX example (8,629 train windows at 131 kb; 1,509 at 1 Mb):

| GPU | Context | Train | Evaluate (2,823 windows) | Insertion 300 sites x 11 motifs |
|---|---|---|---|---|
| A100 80 GB | 131 kb | 26 min (2 epochs) | 4 min | (see below) |
| A100 80 GB | 1 Mb | 61 min (3 epochs) | 27 min | about 2 h |

Add about 2 min per job to load the model. Folds run in parallel, one GPU each.

## Things that went wrong before (already handled, do not undo)

- **Head that never trains.** `prepare_head_specs(..., organism="MUS_MUSCULUS")` in alphagenome_ft gives
  the head one organism slot; mouse is index 1, so the gradients are silently zero and only weight
  decay moves the weights. The pipeline uses `organism=None` (2 slots) and step 02 stops with an error
  if the head did not change. Keep that check.
- **Two bugs in alphagenome_ft 0.1.11** are patched by `env/alphagenome_ft_2c261c4.patch`
  (`_shard_batch` argument; `requested_outputs` default). `install_env.sh` applies it.
- **WSL2 (genos).** JAX fails to allocate GPU memory with the default allocator. Run with
  `AG_ALLOCATOR=cuda_async`. genos has a 16 GB card, so only the 131 kb context fits.
- **Lab scratch can fill up.** `/lustre/fs4/fuchs_lab/scratch` is a 20 TB lab quota and was full in
  Sept 2026; writes then fail silently (truncated files). Check `df -h` on your out_dir before big runs.
- **Pretrained trunk saw all chromosomes.** The `all_folds` weights were trained with ENCODE targets on
  every chromosome, so test numbers are not fully blind. Same caveat for every fold.

## Files

- `config/template.yaml`: commented template. `config/example_sox_cell_lines_atac.yaml`: worked example.
- `agft/config.py` (config reading and checks), `agft/common.py` (data helpers, metrics), `agft/model.py`
  (AlphaGenome helpers).
- `scripts/00..04`: the steps. Each takes `--config` and `--fold`; 02 to 04 take `--smoke`.
- `run_pipeline.sh`, `submit.sh`, `slurm/run.sbatch`, `env.sh`.
- `env/`: `install_env.sh`, `download_weights.sh`, the patch, `requirements.lock.txt` (tested versions:
  python 3.11, jax 0.10.2 CUDA 12, alphagenome_research eecf3d6, alphagenome_ft 0.1.11 2c261c4).
- `folds/mm10/`: chromosome folds.

Origin: cleaned up in October 2026 from the September 2026 AlphaGenome runs of the SOX2/SOX9 project
(`pioneer_factors_scc/results/models/alphagenome` on genos, `HPC_README.md` there has the original
results). The example config uses the same settings, and its fold 0 windows were checked to be
identical to the original 131 kb run (8,629 train, 1,545 valid, 2,823 test). The full rerun was not
repeated; a smoke test of all five steps passed on an L40 on 2026-10-19.
