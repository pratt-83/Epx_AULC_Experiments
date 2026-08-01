# Epiplexity AULC Experiments

This repository contains pilot experiments for measuring epiplexity from genomic
sequence learning dynamics. In this project, epiplexity is treated as a practical
question: how much structure does a sequence region reveal to a small observer,
and how quickly does that observer become better at predicting it?

The current workflow trains the same sequence observer across multiple biological
categories of microbial genomic windows, then compares the validation-loss curves
for each category. Area-under-loss-curve summaries are used as a compact way to
compare how learnable different genomic contexts are over training time.

## Why target non-coding regions?

Protein-coding sequence has strong, well-known constraints: codons, open reading
frames, amino-acid conservation, and selection on translated products. Those
signals are useful, but they can also dominate a learnability experiment because
the model may be rediscovering obvious coding grammar.

Non-coding regions are interesting for the opposite reason. They are not empty
background. Intergenic sequence, RNA genes, pseudogenes, and annotation boundaries
can carry regulatory signals, structural constraints, repeats, mobile-element
signatures, composition biases, and species-specific organization. Much of that
structure is subtler than a protein-coding frame, so it is a good place to ask
whether epiplexity captures learnable biological organization beyond simple
coding status.

The main comparison here is therefore not just "coding versus random." It is a
more graded question: do different annotation contexts produce different
learning trajectories when the observer architecture and training pressure are
held fixed?

## Current experiment

The active microbial annotation pilot uses 512 bp windows assigned to six buckets:

- `CDS_high`
- `intergenic_high`
- `RNA_enriched`
- `pseudogene_enriched`
- `boundary_mixed`
- `mixed_other`

`scripts/make_annotation_buckets.py` samples windows from bacterial FASTA/GFF
inputs and assigns each window to a bucket from annotation overlap. The training
script then expects bucketed train/validation sequence files under
`data/processed/[experiment_id]/real_bucket_files/`.

`scripts/train_observer.py` trains one shared observer while sampling uniformly
across buckets. Validation loss is logged separately for every bucket, making the
loss curves comparable across biological contexts. The supported observers are:

- `transformer`: small causal Transformer baseline
- `hyena`: small causal Hyena language model for longer-range sequence modeling

`scripts/calc_aulc.py` summarizes each bucket's loss curve with:

- initial and final validation loss
- total loss reduction
- half-drop step (`t50_step`)
- raw AULC
- AULC above the final observed loss

## Repository layout

```text
README.md

scripts/
  make_annotation_buckets.py
  train_observer.py
  calc_aulc.py
  plot_early_loss_curves.py

src/
  data_utils.py
  models.py
  hyena_layers.py
  hyena_observer.py
  epiplexity_metrics.py

k8s/
  storage.yaml
  train-transformer.yaml
  train-hyena.yaml
```

Raw data, processed datasets, model checkpoints, result folders, figures, local
notes, and machine-specific files are intentionally kept out of Git.

## Startup reminders

Create annotation metadata:

```bash
python scripts/make_annotation_buckets.py
```

Train a local observer once the bucketed sequence files exist:

```bash
python scripts/train_observer.py \
  --experiment_id microbial_annotation_512bp_v1 \
  --model transformer \
  --seq_len 512
```

Summarize loss curves:

```bash
python scripts/calc_aulc.py \
  --experiment_id microbial_annotation_512bp_v1
```

Plot the early Transformer/Hyena comparison:

```bash
python scripts/plot_early_loss_curves.py \
  --experiment_id microbial_annotation_512bp_v1 \
  --max_step 5000
```

Kubernetes job manifests for the larger Transformer and Hyena runs live in
`k8s/`.
