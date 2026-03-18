#!/usr/bin/env python3

import torch
import torch.nn as nn


# ------------------------------------------------------------
# model definitions
# ------------------------------------------------------------

class TinyCausalTransformer(nn.Module):
	"""
	A small autoregressive transformer observer.

	Structure:
	- token embedding
	- positional embedding
	- stacked transformer encoder blocks used with a causal mask
	- final layer norm
	- linear language modeling head
	"""

	def __init__(
		self,
		vocab_size: int,
		seq_len_in: int,
		d_model: int,
		n_heads: int,
		n_layers: int,
		dropout: float,
	):
		super().__init__()
		self.seq_len_in = seq_len_in

		self.token_embedding = nn.Embedding(vocab_size, d_model)
		self.position_embedding = nn.Embedding(seq_len_in, d_model)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=n_heads,
			dim_feedforward=4 * d_model,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
			norm_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

		self.final_norm = nn.LayerNorm(d_model)
		self.lm_head = nn.Linear(d_model, vocab_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x: Tensor of shape (batch_size, seq_len_in)

		Returns:
			logits: Tensor of shape (batch_size, seq_len_in, vocab_size)
		"""
		batch_size, seq_len = x.shape

		if seq_len != self.seq_len_in:
			raise ValueError(f"Expected input length {self.seq_len_in}, got {seq_len}")

		positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
		hidden = self.token_embedding(x) + self.position_embedding(positions)

		# Prevent each position from attending to future tokens.
		causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

		hidden = self.transformer(hidden, mask=causal_mask)
		hidden = self.final_norm(hidden)
		logits = self.lm_head(hidden)

		return logits

