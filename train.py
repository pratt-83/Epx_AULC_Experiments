#!/usr/bin/env python3
# train.py
#
# PURPOSE
#	Train a small compute-bounded "observer" model and log validation loss curves needed for AULC.
#
# CONTEXT IN PROJECT PLAN
#	This is Phase 1 / Q1: does learnability separate structure vs controls under a fixed observer/budget?
#	- preprocess_data.py created:
#		real_train.txt, shuf_train.txt, rand_train.txt
#		real_val.txt,   shuf_val.txt,   rand_val.txt
#	- This script:
#		(1) trains an observer on a balanced mixture of train sets
#		(2) periodically evaluates loss on each validation set
#		(3) writes a CSV loss curve used by calc_aulc.py
#
# OBSERVER (MODEL) CHOICE
#	A small decoder-only transformer (autoregressive next-token prediction).
#	We keep it simple and fixed; the goal is not SOTA accuracy but stable learning dynamics.
#
# INPUTS
#	--data_dir: directory containing the 6 txt files.
#	Each txt file: one DNA window per line, fixed length = --seq_len.
#	Characters in REAL windows may include N, controls should be A/C/G/T only.
#
# OUTPUTS
#	--log_csv: CSV file with columns:
#		step,loss_real,loss_shuf,loss_rand
#	These are validation losses at each checkpoint step.
#	Optional: logs/tiny_observer.pt checkpoint
#
# CONTROL FLOW SUMMARY
#	1) Load train/val windows for each condition and encode into token IDs.
#	2) Build model + optimizer.
#	3) For each training step:
#		- pick one condition uniformly (real/shuf/rand)
#		- sample a batch of windows from that condition
#		- compute next-token loss and update weights
#	4) Every eval_every steps:
#		- compute mean validation loss on real_val, shuf_val, rand_val
#		- append to log_csv
#
# IMPORTANT DETAIL ABOUT SEQ_LEN
#	Dataset windows have length seq_len (e.g., 2048).
#	Next-token prediction uses:
#		x = tokens[0:seq_len-1]
#		y = tokens[1:seq_len]
#	So the model operates on length (seq_len-1) inputs.

import argparse
import random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Tokenization (character-level)
# ----------------------------
# We use a tiny alphabet: A,C,G,T,N
# This matches your "character-level tokens" decision.
TOKENS = ["A", "C", "G", "T", "N"]
stoi = {c: i for i, c in enumerate(TOKENS)}


def encode_line(line: str) -> torch.Tensor:
	"""
	Convert a DNA string (A/C/G/T/N) into a 1D tensor of integer token IDs.

	Any unknown characters are mapped to 'N'.

	Returns:
		torch.LongTensor of shape (seq_len,)
	"""
	line = line.strip().upper()
	return torch.tensor([stoi.get(c, stoi["N"]) for c in line], dtype=torch.long)


def load_windows(path: Path, expected_len: int) -> List[torch.Tensor]:
	"""
	Load windows from a txt file and encode each line as token IDs.

	Validates:
	- file is not empty
	- each non-empty line has length == expected_len

	Returns:
		List[torch.LongTensor], each of shape (expected_len,)
	"""
	windows: List[torch.Tensor] = []
	with path.open() as f:
		for ln, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			if len(line) != expected_len:
				raise ValueError(f"{path}: line {ln} has length {len(line)} but expected {expected_len}")
			windows.append(encode_line(line))
	if not windows:
		raise ValueError(f"{path}: no windows loaded")
	return windows


# ----------------------------
# Model definition (tiny decoder-only transformer)
# ----------------------------
class TinyCausalTransformer(nn.Module):
	"""
	A small autoregressive transformer observer.

	Implementation approach:
	- Use nn.TransformerEncoder blocks with a causal attention mask.
	- Token embedding + position embedding
	- Stacked attention blocks
	- LayerNorm + linear head to predict next token

	This is intentionally minimal and stable for loss-curve experiments.
	"""

	def __init__(self, vocab_size: int, seq_len_in: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
		"""
		Args:
			vocab_size: size of token alphabet (here 5)
			seq_len_in: length of model input sequence (usually dataset_seq_len - 1)
			d_model: embedding/hidden dimension
			n_heads: number of attention heads
			n_layers: number of transformer layers
			dropout: dropout probability
		"""
		super().__init__()
		self.seq_len_in = seq_len_in

		self.tok_emb = nn.Embedding(vocab_size, d_model)
		self.pos_emb = nn.Embedding(seq_len_in, d_model)

		layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=n_heads,
			dim_feedforward=4 * d_model,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
			norm_first=True,
		)
		self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)

		self.ln_f = nn.LayerNorm(d_model)
		self.lm_head = nn.Linear(d_model, vocab_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass.

		Args:
			x: token IDs, shape (B, T), where T must equal seq_len_in

		Returns:
			logits: shape (B, T, vocab_size)
		"""
		B, T = x.shape
		if T != self.seq_len_in:
			raise ValueError(f"Model expected T={self.seq_len_in}, got T={T}")

		pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

		h = self.tok_emb(x) + self.pos_emb(pos)

		# Causal mask: prevents attending to future positions.
		# True entries are masked.
		mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

		h = self.blocks(h, mask=mask)
		h = self.ln_f(h)
		logits = self.lm_head(h)
		return logits


# ----------------------------
# Batch sampling utilities
# ----------------------------
def sample_batch(examples: List[torch.Tensor], batch_size: int, rng: random.Random) -> torch.Tensor:
	"""
	Sample a batch of full windows from a list of encoded windows.

	Args:
		examples: list of tensors of shape (seq_len,)
		batch_size: number of windows to sample
		rng: Python random.Random instance for reproducibility

	Returns:
		batch tensor of shape (B, seq_len)
	"""
	batch = []
	for _ in range(batch_size):
		batch.append(examples[rng.randrange(len(examples))].unsqueeze(0))
	return torch.cat(batch, dim=0)


@torch.no_grad()
def eval_loss(
	model: nn.Module,
	examples_val: List[torch.Tensor],
	device: str,
	batch_size: int,
	n_batches: int,
	seed: int,
) -> float:
	"""
	Compute mean validation loss on a given validation set by averaging over n_batches sampled batches.

	Why sample batches instead of full pass?
	- Faster for frequent evaluation points
	- Stable enough if n_batches is reasonably large (e.g., 25)

	Args:
		model: trained model
		examples_val: list of validation windows
		device: "cuda" or "cpu"
		batch_size: batch size used for eval
		n_batches: how many random batches to average
		seed: seed for deterministic eval sampling

	Returns:
		mean cross-entropy loss over sampled batches
	"""
	model.eval()
	rng = random.Random(seed)

	losses = []
	for _ in range(n_batches):
		b = sample_batch(examples_val, batch_size, rng).to(device)  # (B, seq_len)
		x = b[:, :-1]  # (B, seq_len-1)
		y = b[:, 1:]   # (B, seq_len-1)
		logits = model(x)  # (B, seq_len-1, vocab)
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
		losses.append(loss.item())

	model.train()
	return float(sum(losses) / len(losses))


def main():
	# ----------------------------
	# Parse CLI arguments
	# ----------------------------
	ap = argparse.ArgumentParser(description="Train observer and log loss curves for AULC.")
	ap.add_argument("--data_dir", required=True, help="Directory containing real/shuf/rand train/val txt files")
	ap.add_argument("--seq_len", type=int, default=2048, help="Length of dataset windows in bp (characters)")
	ap.add_argument("--steps", type=int, default=8000, help="Number of training update steps")
	ap.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
	ap.add_argument("--eval_every", type=int, default=100, help="Evaluate losses every N steps")
	ap.add_argument("--eval_batches", type=int, default=25, help="Number of batches to average per eval loss")
	ap.add_argument("--d_model", type=int, default=512)
	ap.add_argument("--n_heads", type=int, default=8)
	ap.add_argument("--n_layers", type=int, default=6)
	ap.add_argument("--dropout", type=float, default=0.1)
	ap.add_argument("--lr", type=float, default=3e-4)
	ap.add_argument("--seed", type=int, default=1)
	ap.add_argument("--log_csv", default="logs/loss_curves.csv", help="Output CSV path for loss curves")
	ap.add_argument("--save_ckpt", action="store_true", help="If set, save model checkpoint at end")
	args = ap.parse_args()

	# ----------------------------
	# Reproducibility
	# ----------------------------
	random.seed(args.seed)
	torch.manual_seed(args.seed)

	# ----------------------------
	# Load datasets
	# ----------------------------
	data_dir = Path(args.data_dir)

	# Train sets (used for gradient updates)
	real_train = load_windows(data_dir / "real_train.txt", args.seq_len)
	shuf_train = load_windows(data_dir / "shuf_train.txt", args.seq_len)
	rand_train = load_windows(data_dir / "rand_train.txt", args.seq_len)

	# Validation sets (used only to measure loss curves)
	real_val = load_windows(data_dir / "real_val.txt", args.seq_len)
	shuf_val = load_windows(data_dir / "shuf_val.txt", args.seq_len)
	rand_val = load_windows(data_dir / "rand_val.txt", args.seq_len)

	train_sets: List[List[torch.Tensor]] = [real_train, shuf_train, rand_train]

	# ----------------------------
	# Initialize model + optimizer
	# ----------------------------
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# Model input length is seq_len-1 because we predict next token.
	model = TinyCausalTransformer(
		vocab_size=len(TOKENS),
		seq_len_in=args.seq_len - 1,
		d_model=args.d_model,
		n_heads=args.n_heads,
		n_layers=args.n_layers,
		dropout=args.dropout,
	).to(device)

	opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

	# ----------------------------
	# Prepare logging
	# ----------------------------
	log_path = Path(args.log_csv)
	log_path.parent.mkdir(parents=True, exist_ok=True)
	log_path.write_text("step,loss_real,loss_shuf,loss_rand\n")

	rng = random.Random(args.seed)

	# ----------------------------
	# Training loop
	# ----------------------------
	for step in range(1, args.steps + 1):
		# Balanced condition selection:
		# Each step we randomly choose which dataset to sample from, uniformly among the 3 conditions.
		which = rng.randrange(3)
		examples = train_sets[which]

		# Sample a batch of full windows: shape (B, seq_len)
		b = sample_batch(examples, args.batch_size, rng).to(device)

		# Next-token prediction setup:
		# x: tokens 0..(seq_len-2) -> length (seq_len-1)
		# y: tokens 1..(seq_len-1) -> length (seq_len-1)
		x = b[:, :-1]
		y = b[:, 1:]

		logits = model(x)
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

		opt.zero_grad(set_to_none=True)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		opt.step()

		# ----------------------------
		# Evaluation: loss curves for AULC
		# ----------------------------
		if step == 1 or step % args.eval_every == 0:
			# We use different seeds per step so the eval sampling changes each time,
			# but is still deterministic/reproducible.
			lr = eval_loss(model, real_val, device, args.batch_size, args.eval_batches, args.seed + 1000 + step)
			ls = eval_loss(model, shuf_val, device, args.batch_size, args.eval_batches, args.seed + 2000 + step)
			ln = eval_loss(model, rand_val, device, args.batch_size, args.eval_batches, args.seed + 3000 + step)

			with log_path.open("a") as f:
				f.write(f"{step},{lr:.6f},{ls:.6f},{ln:.6f}\n")

			print(f"step {step:>5} | train_loss {loss.item():.4f} | val real {lr:.4f} shuf {ls:.4f} rand {ln:.4f}")

	# ----------------------------
	# Optional: save checkpoint
	# ----------------------------
	if args.save_ckpt:
		Path("logs").mkdir(parents=True, exist_ok=True)
		torch.save(model.state_dict(), "logs/tiny_observer.pt")
		print("Saved checkpoint to logs/tiny_observer.pt")

	print("Done. Loss curve CSV written to:", log_path)


if __name__ == "__main__":
	main()

