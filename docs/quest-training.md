# Training PolyUMI policies on Quest

The Quest checkout prepared for this project is:

```text
/projects/p52914/xph8283/polyumi/PolyUMI
```

The DP and Vista environments live beside it under `envs/`. Datasets, shared model caches,
outputs, generated training matrices, and Slurm logs also live under
`/projects/p52914/xph8283/polyumi/`.

## Upload one dataset

Export the dataset locally before training:

- DP uses a normal `pingest export` / `--type dp` export.
- Vista requires `pingest export --type polyumi`, which includes wrist vision, finger vision,
  contact audio, and proprioception. A PolyUMI export can also be used by DP; its loader ignores
  the additional streams.

From a local Ubuntu terminal, copy the exported archive to Quest:

```bash
scp /absolute/local/path/task_a.zarr.zip \
  xph8283@login.quest.northwestern.edu:/projects/p52914/xph8283/polyumi/datasets/
```

All dataset paths passed to the training scripts must be absolute Quest paths.

## Run one dataset/model pair

Log into Quest and move to the checkout:

```bash
ssh xph8283@login.quest.northwestern.edu
cd /projects/p52914/xph8283/polyumi/PolyUMI
```

Run a two-epoch DP smoke test with batch size 8:

```bash
sbatch scripts/quest/train_one.sbatch \
  dp \
  /projects/p52914/xph8283/polyumi/datasets/task_a.zarr.zip
```

Run a two-epoch Vista smoke test:

```bash
sbatch scripts/quest/train_one.sbatch \
  vista \
  /projects/p52914/xph8283/polyumi/datasets/task_a_multimodal.zarr.zip \
  vista
```

The optional fourth and fifth arguments set epochs and batch size. This example requests 120
epochs with a batch size of 8:

```bash
sbatch --time=24:00:00 scripts/quest/train_one.sbatch \
  vista /projects/p52914/xph8283/polyumi/datasets/task_a_multimodal.zarr.zip \
  vista 120 8
```

The single-job script defaults to one A100, 8 CPU cores, 64 GB of RAM, and a two-hour limit.
Any `sbatch` options placed before the script path override those defaults.
Submit it from the repository root as shown above. If submitting from elsewhere, export
`POLYUMI_REPO_ROOT=/projects/p52914/xph8283/polyumi/PolyUMI` first.

Monitor jobs and read their logs with:

```bash
squeue -u xph8283
tail -f /projects/p52914/xph8283/polyumi/logs/polyumi-one-JOB_ID.out
```

Each run writes to `outputs/DATASET_NAME/MODEL_NAME/JOB_ID/`. A completed run contains a
`SUCCESS` marker alongside its Hydra output and checkpoints.

## Run many datasets and models efficiently

The matrix submitter creates the compatible Cartesian product of two tab-separated files and
submits it as one Slurm array. Copy the templates:

```bash
cp config/quest/datasets.example.tsv datasets.tsv
cp config/quest/models.example.tsv models.tsv
```

`datasets.tsv` has three columns:

```text
name    path                                                              type
task_a  /projects/p52914/xph8283/polyumi/datasets/task_a.zarr.zip          dp
task_b  /projects/p52914/xph8283/polyumi/datasets/task_b.zarr.zip          polyumi
```

Use `dp` for a vision/proprioception export and `polyumi` for a multimodal Vista export. Names
become output-directory components, so use letters, numbers, dots, underscores, or hyphens.

`models.tsv` has five columns:

```text
name            policy  variant                                          epochs  batch_size
dp_timm         dp      train_diffusion_unet_timm_polyumi_workspace      120     32
polytouch       vista   polytouch                                        120     4
see_hear_feel   vista   see_hear_feel                                    120     8
sparsh_x        vista   sparsh_x                                         120     8
vista           vista   vista                                            120     8
```

Vista variants are `polytouch`, `see_hear_feel`, `sparsh_x`, and `vista`. These correspond to
the four model-specific YAML files under `external/polyumi_vista_policy/vista/config/`. A DP
variant is the name of a YAML file under
`external/polyumi_diffusion_policy/diffusion_policy/config/`, without `.yaml`.

Preview and validate the generated combinations:

```bash
scripts/quest/submit_training_matrix.sh \
  --datasets datasets.tsv --models models.tsv --dry-run
```

Submit the array, allowing at most two simultaneous GPU jobs:

```bash
scripts/quest/submit_training_matrix.sh \
  --datasets datasets.tsv \
  --models models.tsv \
  --max-parallel 2 \
  --time 24:00:00
```

DP models run on both dataset types. Vista models are automatically paired only with `polyumi`
datasets. Each array element trains one pair, so one failure does not stop the other experiments.
The concurrency cap controls GPU use and shared-filesystem pressure; increase it only when the
allocation and dataset I/O can support more simultaneous runs.

For online Weights & Biases logging, authenticate without placing the key in these files, then
submit with `--wandb-mode online`:

```bash
export WANDB_API_KEY='your-key'
scripts/quest/submit_training_matrix.sh \
  --datasets datasets.tsv --models models.tsv --wandb-mode online
```

The generated matrix is retained under `manifests/generated/`, giving every array job an exact
record of the dataset, model, epoch count, and batch size it received.
