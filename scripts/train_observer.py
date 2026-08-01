#!/usr/bin/env python3

import argparse
import random
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from src.hyena_observer import TinyCausalHyenaLM
from src.models import TinyCausalTransformer


TOKENS = ["A", "C", "G", "T", "N"]
TOKEN_TO_ID = {token: i for i, token in enumerate(TOKENS)}


# ------------------------------------------------------------
# sequence loading helpers
# ------------------------------------------------------------

def encode_line(line: str) -> torch.Tensor:
	"""
	Convert one DNA string into token IDs.

	Unknown characters are mapped to N.
	"""
	line = line.strip().upper()
	token_ids = [TOKEN_TO_ID.get(base, TOKEN_TO_ID["N"]) for base in line]
	return torch.tensor(token_ids, dtype=torch.long)


def load_windows(path: Path, expected_len: int) -> List[torch.Tensor]:
	"""
	Load one sequence window per line and encode each as token IDs.
	"""
	windows: List[torch.Tensor] = []

	with path.open() as handle:
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue

			if len(line) != expected_len:
				raise ValueError(
					f"{path}: line {line_number} has length {len(line)} but expected {expected_len}"
				)

			windows.append(encode_line(line))

	if not windows:
		raise ValueError(f"{path}: no windows loaded")

	return windows


def sample_batch(examples: List[torch.Tensor], batch_size: int, rng: random.Random) -> torch.Tensor:
	"""
	Sample a batch of full windows with replacement.
	"""
	batch = []

	for _ in range(batch_size):
		example = examples[rng.randrange(len(examples))]
		batch.append(example.unsqueeze(0))

	return torch.cat(batch, dim=0)


# ------------------------------------------------------------
# evaluation helper
# ------------------------------------------------------------

@torch.no_grad()
def eval_loss(
	model: torch.nn.Module,
	examples: List[torch.Tensor],
	device: str,
	batch_size: int,
	n_batches: int,
	seed: int,
) -> float:
	"""
	Estimate mean validation loss from sampled batches.

	This keeps evaluation cost predictable even when validation sets grow.
	"""
	model.eval()
	rng = random.Random(seed)

	losses = []

	for _ in range(n_batches):
		batch = sample_batch(examples, batch_size, rng).to(device)
		x = batch[:, :-1]
		y = batch[:, 1:]

		logits = model(x)
		loss = F.cross_entropy(
			logits.reshape(-1, logits.size(-1)),
			y.reshape(-1),
		)
		losses.append(loss.item())

	model.train()
	return float(sum(losses) / len(losses))


# ------------------------------------------------------------
# main training entrypoint
# ------------------------------------------------------------

def main() -> None:
	parser = argparse.ArgumentParser(
		description="Train a shared observer and log condition-specific validation loss curves."
	)
	parser.add_argument(
		"--experiment_id",
		required=True,
		help="Experiment tag used for processed data and outputs",
	)
	parser.add_argument(
		"--processed_root",
		default="data/processed",
		help="Root directory containing data/processed/[experiment_id]/",
	)
	parser.add_argument(
		"--outputs_root",
		default="outputs",
		help="Root directory containing outputs/[experiment_id]/",
	)
	parser.add_argument(
		"--seq_len",
		type=int,
		default=512,
		help="Dataset window length",
	)
	parser.add_argument(
		"--steps",
		type=int,
		default=8000,
		help="Number of training steps",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=16,
		help="Batch size",
	)
	parser.add_argument(
		"--eval_every",
		type=int,
		default=100,
		help="Evaluate every N training steps",
	)
	parser.add_argument(
		"--eval_batches",
		type=int,
		default=25,
		help="Number of sampled validation batches per evaluation",
	)
	parser.add_argument("--d_model", type=int, default=512)
	parser.add_argument(
		"--model",
		choices=("transformer", "hyena"),
		default="transformer",
		help="Observer architecture to train",
	)
	parser.add_argument("--n_heads", type=int, default=8)
	parser.add_argument("--n_layers", type=int, default=6)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--hyena_order", type=int, default=2)
	parser.add_argument("--hyena_filter_order", type=int, default=64)
	parser.add_argument(
		"--hyena_use_abs_pos",
		action="store_true",
		help="If set, add absolute positional embeddings to the Hyena observer",
	)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument(
		"--save_ckpt",
		action="store_true",
		help="If set, save a checkpoint at the end of training",
	)
	args = parser.parse_args()

	random.seed(args.seed)
	torch.manual_seed(args.seed)

	data_dir = Path(args.processed_root) / args.experiment_id / "real_bucket_files"
	output_dir = Path(args.outputs_root) / args.experiment_id
	log_dir = output_dir / "logs"
	ckpt_dir = output_dir / "ckpt"

	log_dir.mkdir(parents=True, exist_ok=True)
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	log_csv_path = log_dir / "loss_curves.csv"
	ckpt_path = ckpt_dir / f"{args.model}_observer.pt"

	CDS_high_train = load_windows(
		data_dir / "train_CDS_high_real.txt",
		args.seq_len,
	)
	intergenic_high_train = load_windows(
		data_dir / "train_intergenic_high_real.txt",
		args.seq_len,
	)
	RNA_enriched_train = load_windows(
		data_dir / "train_RNA_enriched_real.txt",
		args.seq_len,
	)
	pseudogene_enriched_train = load_windows(
		data_dir / "train_pseudogene_enriched_real.txt",
		args.seq_len,
	)
	boundary_mixed_train = load_windows(
		data_dir / "train_boundary_mixed_real.txt",
		args.seq_len,
	)
	mixed_other_train = load_windows(
		data_dir / "train_mixed_other_real.txt",
		args.seq_len,
	)

	CDS_high_val = load_windows(
		data_dir / "val_CDS_high_real.txt",
		args.seq_len,
	)
	intergenic_high_val = load_windows(
		data_dir / "val_intergenic_high_real.txt",
		args.seq_len,
	)
	RNA_enriched_val = load_windows(
		data_dir / "val_RNA_enriched_real.txt",
		args.seq_len,
	)
	pseudogene_enriched_val = load_windows(
		data_dir / "val_pseudogene_enriched_real.txt",
		args.seq_len,
	)
	boundary_mixed_val = load_windows(
		data_dir / "val_boundary_mixed_real.txt",
		args.seq_len,
	)
	mixed_other_val = load_windows(
		data_dir / "val_mixed_other_real.txt",
		args.seq_len,
	)

	train_sets: List[List[torch.Tensor]] = [
		CDS_high_train,
		intergenic_high_train,
		RNA_enriched_train,
		pseudogene_enriched_train,
		boundary_mixed_train,
		mixed_other_train,
	]

	device = "cuda" if torch.cuda.is_available() else "cpu"

	if args.model == "transformer":
		model = TinyCausalTransformer(
			vocab_size=len(TOKENS),
			seq_len_in=args.seq_len - 1,
			d_model=args.d_model,
			n_heads=args.n_heads,
			n_layers=args.n_layers,
			dropout=args.dropout,
		)
	else:
		model = TinyCausalHyenaLM(
			vocab_size=len(TOKENS),
			seq_len_in=args.seq_len - 1,
			d_model=args.d_model,
			n_layers=args.n_layers,
			dropout=args.dropout,
			hyena_order=args.hyena_order,
			hyena_filter_order=args.hyena_filter_order,
			use_abs_pos=args.hyena_use_abs_pos,
		)

	model = model.to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	print(f"Training model={args.model} on device={device}")

	log_csv_path.write_text(
		"step,"
		"loss_CDS_high,"
		"loss_intergenic_high,"
		"loss_RNA_enriched,"
		"loss_pseudogene_enriched,"
		"loss_boundary_mixed,"
		"loss_mixed_other\n"
	)
	rng = random.Random(args.seed)

	for step in range(1, args.steps + 1):
		# Sample one of the six conditions uniformly so each receives
		# comparable optimization pressure over the run.
		condition_index = rng.randrange(6)
		examples = train_sets[condition_index]

		batch = sample_batch(examples, args.batch_size, rng).to(device)
		x = batch[:, :-1]
		y = batch[:, 1:]

		logits = model(x)
		train_loss = F.cross_entropy(
			logits.reshape(-1, logits.size(-1)),
			y.reshape(-1),
		)

		optimizer.zero_grad(set_to_none=True)
		train_loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()

		if step == 1 or step % args.eval_every == 0:
			# Validation is computed separately for each condition even though
			# training uses a shared observer.
			loss_CDS_high = eval_loss(
				model=model,
				examples=CDS_high_val,
				device=device,
				batch_size=args.batch_size,
				n_batches=args.eval_batches,
				seed=args.seed + 1000 + step,
			)
			loss_intergenic_high = eval_loss(
				model=model,
				examples=intergenic_high_val,
				device=device,
				batch_size=args.batch_size,
				n_batches=args.eval_batches,
				seed=args.seed + 2000 + step,
			)
			loss_RNA_enriched = eval_loss(
				model=model,
				examples=RNA_enriched_val,
				device=device,
				batch_size=args.batch_size,
				n_batches=args.eval_batches,
				seed=args.seed + 3000 + step,
			)
			loss_pseudogene_enriched = eval_loss(
				model=model,
				examples=pseudogene_enriched_val,
				device=device,
				batch_size=args.batch_size,
				n_batches=args.eval_batches,
				seed=args.seed + 4000 + step,
			)
			loss_boundary_mixed = eval_loss(
				model=model,
				examples=boundary_mixed_val,
				device=device,
				batch_size=args.batch_size,
				n_batches=args.eval_batches,
				seed=args.seed + 5000 + step,
			)
			loss_mixed_other = eval_loss(
				model=model,
				examples=mixed_other_val,
				device=device,
				batch_size=args.batch_size,
				n_batches=args.eval_batches,
				seed=args.seed + 6000 + step,
			)

			with log_csv_path.open("a") as handle:
				handle.write(
					f"{step},"
					f"{loss_CDS_high:.6f},"
					f"{loss_intergenic_high:.6f},"
					f"{loss_RNA_enriched:.6f},"
					f"{loss_pseudogene_enriched:.6f},"
					f"{loss_boundary_mixed:.6f},"
					f"{loss_mixed_other:.6f}\n"
				)

			print(
				f"step {step:>5} | "
				f"train_loss {train_loss.item():.4f} | "
				f"val CDS_high {loss_CDS_high:.4f} | "
				f"val intergenic_high {loss_intergenic_high:.4f} | "
				f"val RNA_enriched {loss_RNA_enriched:.4f} | "
				f"val pseudogene_enriched {loss_pseudogene_enriched:.4f} | "
				f"val boundary_mixed {loss_boundary_mixed:.4f} | "
				f"val mixed_other {loss_mixed_other:.4f}"
			)

	if args.save_ckpt:
		torch.save(model.state_dict(), ckpt_path)
		print(f"Saved checkpoint to: {ckpt_path}")

	print(f"Loss curve CSV written to: {log_csv_path}")


if __name__ == "__main__":
	main()
	