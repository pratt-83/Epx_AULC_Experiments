# Data Layout

This repository does not track raw or processed datasets.

Expected layout:

```text
data/
	raw/
		[fasta files and subdirectories]
	processed/
		[experiment_id]/
			real_train.txt
			real_val.txt
			shuf_train.txt
			shuf_val.txt
			rand_train.txt
			rand_val.txt
			meta.tsv
```

Notes:

- Put raw `.fna` inputs under `data/raw/`.
- Run `scripts/preprocess_data.py` to create processed datasets under `data/processed/[experiment_id]/`.
- Processed datasets are not committed to Git.
