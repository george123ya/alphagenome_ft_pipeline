# Reproduce from scratch on a new machine

Every step to go from an empty Linux machine with an NVIDIA GPU to the SOX cell line results in
`README.md`. The pipeline code is in this repo; three things are not, and each has its own step below:

| What | Size | Why it is not in the repo | Step |
|---|---|---|---|
| Python environment (JAX, alphagenome_research, alphagenome_ft) | about 6 GB | built from pinned versions | 3 |
| AlphaGenome weights (`all_folds`) | about 700 MB | personal licence, cannot be redistributed | 4 |
| Input data: mm10 FASTA, ATAC bigwigs, peaks, contaminant BED | about 3 GB | FASTA is public; the rest is unpublished lab data | 5 |

Needs: Linux (or WSL2), an NVIDIA GPU with at least 16 GB for the 131 kb context (28 GB for 1 Mb),
a recent NVIDIA driver (CUDA 12), git, and about 15 GB of disk.

## 1. Get the code

    git clone https://github.com/george123ya/alphagenome_ft_pipeline.git
    cd alphagenome_ft_pipeline

## 2. Conda (skip if you have it)

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    export PATH=$HOME/miniconda3/bin:$PATH

## 3. Environment (about 10 min)

`env/install_env.sh` creates a python 3.11 env, clones the two AlphaGenome repos at the tested commits,
applies our patch to alphagenome_ft, and installs everything pinned to `env/requirements.lock.txt`:

    bash env/install_env.sh $HOME/envs/alphagenome
    echo 'export AG_ENV=$HOME/envs/alphagenome' >> ~/.bashrc
    export AG_ENV=$HOME/envs/alphagenome

It ends with `imports ok, jax 0.10.2 [CudaDevice(id=0)]` and `ENV_DONE`. If it lists only a CPU
device, the NVIDIA driver is missing or too old.

The two upstream repos it fetches (for reference; the script does this for you):

| Repo | Commit |
|---|---|
| https://github.com/google-deepmind/alphagenome_research | `eecf3d60234dda0428f5f083d22e518b72487ce0` |
| https://github.com/genomicsxai/alphagenome_ft (version 0.1.11) | `2c261c48225c6c5d9c1185529b24517b82d8d174` + `env/alphagenome_ft_2c261c4.patch` |

To do it by hand:

    git clone https://github.com/genomicsxai/alphagenome_ft.git
    git -C alphagenome_ft checkout 2c261c48225c6c5d9c1185529b24517b82d8d174
    git -C alphagenome_ft apply /path/to/alphagenome_ft_pipeline/env/alphagenome_ft_2c261c4.patch

## 4. AlphaGenome weights (about 700 MB)

Once per person:

1. Log in at https://huggingface.co and open https://huggingface.co/google/alphagenome-all-folds
2. Accept the model terms (non-commercial use only; access is granted right away).
3. Make a read token at https://huggingface.co/settings/tokens
4. `$AG_ENV/bin/hf auth login` and paste the token.

Then:

    bash env/download_weights.sh $HOME/alphagenome_data/weights/all_folds

## 5. Input data

Put everything in one folder, for example `$HOME/alphagenome_data`:

    D=$HOME/alphagenome_data; mkdir -p $D/genome $D/bw $D/peaks $D/beds

**mm10 genome (public, about 2.7 GB unpacked):**

    wget -O $D/genome/mm10.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz
    gunzip $D/genome/mm10.fa.gz

The pipeline makes the `.fai` index itself on first use.

**Lab data (not public; copy it from where it lives).** On the RU HPC it is under
`/lustre/fs4/fuchs_lab/scratch/gmunoz/pioneer_factors_scc_models/shared/`, and on genos under
`~/gmunoz/SCRATCH/Projects/pioneer_factors_scc/`:

| File | RU HPC (`.../shared/`) | genos (`pioneer_factors_scc/`) | md5 |
|---|---|---|---|
| ATAC_Sox9hi.bw | `bw/` | `figures/carol_comparison_2026-09/tracks/data/bw/` | `c5cb98fd67fb303584301c01e9c908ba` |
| ATAC_Double.bw | `bw/` | same | `0f4afe65298ff6c3dfff88715b446d0f` |
| ATAC_Sox2hi.bw | `bw/` | same | `9ecc079589b6b241640e39a6dacb6ba4` |
| Sox9hi.IDR_0.05.PooledInRep1AndRep2.filt.narrowPeak.gz | `peaks_idr/` | `results/atac/for_R/peaks_idr/` | `5f22f0a4c176ad4adf1468688ee6e01e` |
| Double.IDR_0.05.PooledInRep1AndRep2.filt.narrowPeak.gz | `peaks_idr/` | same | `3da84244d0403c1e16cb08780c2eebf4` |
| Sox2hi.IDR_0.05.PooledInRep1AndRep2.filt.narrowPeak.gz | `peaks_idr/` | same | `87881fbd9f29b4e4777e2b9c1cffc6c2` |
| contaminant_loci.bed | `beds/` | `figures/carol_comparison_2026-09/contamination/tables/` | `72c621e6c70fb71ec4dafc586f58ba6d` |

The md5 values are of the genos copies. For example, from the HPC:

    S=YOUR_USER@HPC_LOGIN_HOST:/lustre/fs4/fuchs_lab/scratch/gmunoz/pioneer_factors_scc_models/shared
    rsync -av "$S/bw/ATAC_{Sox9hi,Double,Sox2hi}.bw" $D/bw/
    rsync -av "$S/peaks_idr/{Sox9hi,Double,Sox2hi}.IDR_0.05.PooledInRep1AndRep2.filt.narrowPeak.gz" $D/peaks/
    rsync -av "$S/beds/contaminant_loci.bed" $D/beds/
    md5sum $D/bw/* $D/peaks/* $D/beds/*      # compare with the table

## 6. Point the config at your copies

    cp config/example_sox_cell_lines_atac.yaml config/sox_local.yaml

In `config/sox_local.yaml`, replace `out_dir`, `genome_fasta`, `weights`, the three `bigwig:` paths,
the three `peaks` and `exclude_bed` with your paths from steps 4 and 5. Leave everything else as it is;
those are the settings behind the published numbers. A one-liner that does it if you used
`$HOME/alphagenome_data`:

    D=$HOME/alphagenome_data
    sed -i -e "s#/lustre/fs4/fuchs_lab/scratch/gmunoz/alphagenome_runs#$D/runs#" \
           -e "s#/lustre/.*/genome/mm10.fa#$D/genome/mm10.fa#" \
           -e "s#/lustre/.*/weights/all_folds#$D/weights/all_folds#" \
           -e "s#/lustre/.*/shared/bw/#$D/bw/#" \
           -e "s#/lustre/.*/shared/peaks_idr/#$D/peaks/#" \
           -e "s#/lustre/.*/shared/beds/#$D/beds/#" config/sox_local.yaml
    grep -n lustre config/sox_local.yaml     # should print nothing

## 7. Run

Without SLURM (one GPU, one fold at a time). On WSL2 add `AG_ALLOCATOR=cuda_async`:

    STEPS="00 01" bash run_pipeline.sh config/sox_local.yaml 0      # checks + windows, CPU, 2 min
    SMOKE=1 bash run_pipeline.sh config/sox_local.yaml 0            # 10 to 15 min, must end PIPELINE_DONE
    bash run_pipeline.sh config/sox_local.yaml 0                    # full fold 0, about 1 h at 131 kb
    for f in 1 2 3 4; do bash run_pipeline.sh config/sox_local.yaml $f; done

With SLURM, `submit.sh` sends one job per fold; the partition names in `slurm/run.sbatch` and
`submit.sh` are the RU HPC ones, so change them for another cluster:

    GPU=l40 bash submit.sh config/sox_local.yaml 0 1 2 3 4

## 8. Check you got the same thing

- Step 01, fold 0: `<out_dir>/fold_0/windows/*.stats.txt` must show **8,629 train, 1,545 valid,
  2,823 test** windows. These were checked to match the original run.
- Step 02: `train_info.json` must show the head weights changed (well above 1e-4).
- Step 03, fold 0, `test_metrics.tsv`: per-sample `peak_log1p_pearson` 0.68 to 0.77,
  Sox2hi vs Sox9hi `peak_logratio_pearson` about 0.43, Double vs Sox9hi about 0.1.

Training on GPU is not bit-for-bit deterministic, so expect the metrics to differ in the second
decimal place, not more.
