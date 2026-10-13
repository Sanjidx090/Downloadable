# Experiment C — 4-source causal Zipformer fine-tuning

## Scope and status

This package was prepared using the known SAPC2 training layout and the original
Icefall fine-tuning resume interfaces from earlier project logs. The user's
**actual `finetune_original.py` has not been uploaded**, so the patcher is
**conservative**: it modifies exactly the two identifiable SAPC2 manifest load
statements and aborts if they cannot be found. The package is syntax-tested and
has offline miniature fixture tests; **it has not yet passed a GPU smoke test**.

It does **not** overwrite the existing `finetune.py` or checkpoint files.

Three execution modes:

* `start`: initialize *model weights only* from `epoch-6.pt`, `epoch-8.pt`, or `epoch-19.pt`; optimizer and scheduler reset for the new run.
* `resume`: restore epoch or batch checkpoint **from this Experiment C folder** with Icefall's normal model, optimizer, scheduler, and available sampler state, without changing the manifest.
* `restart`: initialize model weights from any checkpoint but begin **a new experiment** in a **new folder** with a fresh optimizer/scheduler.

Checkpoint files are those produced by Icefall itself (`epoch-N.pt` and
`checkpoint-B.pt`). This package does **not** create or claim fake `latest.pt`
or `best.pt` aliases.

## 1. Copy these files from Windows downloads to WSL

If ZIP was downloaded into Windows Downloads:

```bash
mkdir -p ~/croko_training/experiment_c
cd ~/croko_training/experiment_c
unzip -o /mnt/c/Users/<WINDOWS_USERNAME>/Downloads/zipformer_experiment_c.zip
```

Or copy the ZIP anywhere you prefer and unzip in that directory.

```bash
source ~/miniconda3/bin/activate zipformer
```

## 2. Audit SAPC2 speaker IDs (not a full lineage audit)

```bash
python audit_speakers.py
```

Requires `/mnt/g/SAPC2_Dataset/manifests/sapc2_cuts_dev.jsonl.gz`.
If your actual full Dev path differs, use `--dev /ACTUAL/DEV_MANIFEST`.
**Never** ignore a positive speaker intersection. Confirm the SAPC2
augmentation was created only from SAPC2 Train original source recordings.

## 3. Prepare exact 70/20/7/3 mix by CUT count

Preview only:

```bash
python prepare_mix.py --plan
```

Build the **120,000-cut pilot** (84k original, 24k SAPC2 aug,
8.4k Croko, 3.6k TORGO). No audio WAV is copied.

```bash
python prepare_mix.py \
  --total-cuts 120000 \
  --out /mnt/d/Croko_Training/manifests/experiment_c_mix_120k.jsonl.gz
```

Each Lhotse cut gets a unique ID. TORGO can repeat at most **twice** in
larger mixes. This finite, precomputed manifest gives an exact mixture
**by number of cuts**, not audio duration or gradient contribution. It is
seeded and immutable within a training run, allowing normal Icefall resume.
It is a **fixed sample**: to expose more SAPC2 examples in full training,
build a **new manifest** and start a **new run**, rather than replacing the
manifest in an existing experiment.

For a broader full experiment after the pilot, create a new 300k-cut mix:

```bash
python prepare_mix.py --total-cuts 300000 --seed 43 \
  --out /mnt/d/Croko_Training/manifests/experiment_c_mix_300k_seed43.jsonl.gz
```

This yields 210k SAPC2 original, 60k SAPC2 aug, 21k Croko and
9k TORGO (TORGO reused <= 2 times). One training epoch is a sweep
through these selected cuts, not all original SAPC2 training data.

## 4. Patch the local causal Zipformer trainer safely

```bash
python patch_trainer.py \
  --source ~/icefall/egs/librispeech/ASR/zipformer/finetune.py \
  --output ~/icefall/egs/librispeech/ASR/zipformer/finetune_4way.py
```

If the two assignments have changed since earlier runs, this step
fails with a message to upload `finetune_original.py`. Do not bypass the
safety check by replacing random regions of the code.

## 5. Preflight / dry run (no training)

```bash
python run_experiment_c.py start --dry-run \
  --init-checkpoint /mnt/g/SAPC2_Dataset/exp_finetune_causal/epoch-6.pt \
  --exp-dir /mnt/d/Croko_Training/exp_4way_C_epoch6_pilot \
  --epochs 2
```

Check that the command targets the causal script and the right paths.

## 6. Short real-GPU smoke test (recommended before full run)

Use a **separate** experiment folder and short checkpoint interval so you
can stop after 20-40 training batches without losing progress:

```bash
python run_experiment_c.py start \
  --init-checkpoint /mnt/g/SAPC2_Dataset/exp_finetune_causal/epoch-6.pt \
  --exp-dir /mnt/d/Croko_Training/exp_4way_C_SMOKE \
  --epochs 1 --max-duration 80 \
  --save-every-n 20 --average-period 20
```

Wait until `checkpoint-20.pt` appears and has finished writing, then
press Ctrl+C. Check the latest log for OOM and train loss. A healthy
smoke run does **not** establish WER improvement.

The smoke test can be kept or removed later. It should **not** be
resumed as the full experiment because the smaller max-duration,
different averaging period and short checkpoint interval must remain
fixed for reproducible restoration.

## 7. Begin full training

```bash
python run_experiment_c.py start \
  --init-checkpoint /mnt/g/SAPC2_Dataset/exp_finetune_causal/epoch-6.pt \
  --exp-dir /mnt/d/Croko_Training/exp_4way_C_epoch6_pilot \
  --epochs 2
```

This initializes model weights from epoch 6, starts a **new** optimizer,
scheduler, epoch counter, and output directory.

Default training settings: causal, chunk sizes 16/32 and left context
128/256, `--base-lr 0.0005`, `--max-duration 120`, FP16, 4 workers,
`--save-every-n 1000`, model averaging period 200. These are trial
parameters; if CUDA OOM occurs, start a **new run** with
`--max-duration 80` or smaller, rather than silently changing
hyperparameters during a resume.

The default validation data is the previously used SAPC2 small real dev
manifest. Keep the full Dev set for final evaluation.

## 8. Resume after interruption

Auto-detect latest valid saved batch/epoch checkpoint in Experiment C:

```bash
python run_experiment_c.py resume \
  --exp-dir /mnt/d/Croko_Training/exp_4way_C_epoch6_pilot
```

Choose an exact checkpoint in that folder:

```bash
python run_experiment_c.py resume \
  --exp-dir /mnt/d/Croko_Training/exp_4way_C_epoch6_pilot \
  --checkpoint epoch-1.pt
```

Or `--checkpoint checkpoint-1000.pt` for step checkpoint.
**Batch checkpoints are best-effort mid-epoch resume**: Icefall's
sampler must successfully restore its saved state; if that fails,
resume from the most recent complete `epoch-N.pt` instead.

You may extend total number of epochs:

```bash
python run_experiment_c.py resume \
  --exp-dir /mnt/d/Croko_Training/exp_4way_C_epoch6_pilot \
  --epochs 4
```

**Important:** Checkpoint selection by modified time normally finds
latest. If files were copied, modified times might not represent
training order; explicitly pass `--checkpoint`.

## 9. Restart from epoch 19 or a chosen Experiment C checkpoint

```bash
python run_experiment_c.py restart \
  --checkpoint /mnt/g/SAPC2_Dataset/exp_finetune_causal/epoch-19.pt \
  --exp-dir /mnt/d/Croko_Training/exp_4way_C_epoch19_compare \
  --epochs 2
```

This **does not restore** optimizer state from epoch 19; it is a new
experiment initialized from epoch-19 weights. To use a different
manifest in a new run, add `--train-manifest /PATH/TO/NEW_MIX.jsonl.gz`.

Inspect saved files:

```bash
python run_experiment_c.py status --exp-dir /mnt/d/Croko_Training/exp_4way_C_epoch6_pilot
```

## Notes / safety limitations

- There is no guarantee that epoch-6 is the best checkpoint. Compare its
  WER/CER with epoch-8/19 using the same real-speaker validation set.
- Croko is **fluent TTS**, not authentic dysarthric speech; limit its
  weight until it demonstrates improvements on real dysarthric Dev.
- Dataset percentages are **by cut count**, not hours, and batch ordering
  remains governed by Icefall's sampler.
- Reusing TORGO utterances at most twice still can overfit only 8
  dysarthric speakers; assess speaker-level WER.
- `resume` requires an unchanged manifest and unchanged CLI training
  hyperparameters recorded at launch; validation manifests are also hashed.
- Local Icefall itself decides exact saved optimizer/scaler/sampler state.
  This wrapper uses native Icefall checkpointing; it does not guarantee
  bit-identical mid-epoch restoration across all versions.
- New experiment checkpoints are on D:, older epoch checkpoints are on G:.
- External-data competition eligibility must be checked separately.

## Troubleshooting

**Patch script cannot locate training assignment:** Upload the user's
local `D:\\Croko_Training\\finetune_original.py` to ChatGPT. The patcher
will not guess with an unknown codebase.

**Missing Dev manifest:** Use the actual file path with
`run_experiment_c.py start --valid-manifest PATH`.

**Torch checkpoint refused as non-ZIP:** Your checkpoint may use legacy
PyTorch serialization. Upload traceback/checkpoint metadata for safe
handling, rather than bypassing validation blindly.

**Unexpected high validation loss:** Check real-speaker validation and
Croko's transcript/prosody quality, and run a real-only control.
