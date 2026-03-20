#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hyena_layers import HyenaOperator


# ------------------------------------------------------------
# model components
# ------------------------------------------------------------

class FeedForward(nn.Module):
	"""
	Standard language-model MLP: D -> 4D -> D.
	"""

	def __init__(self, d_model: int, dropout: float, mult: int = 4):
		super().__init__()
		inner_dim = mult * d_model

		self.fc1 = nn.Linear(d_model, inner_dim)
		self.fc2 = nn.Linear(inner_dim, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.fc1(x)
		x = F.gelu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x


class HyenaBlock(nn.Module):
	"""
	Pre-norm residual block:
	- x = x + HyenaOperator(LN(x))
	- x = x + MLP(LN(x))
	"""

	def __init__(
		self,
		d_model: int,
		l_max: int,
		dropout: float,
		hyena_order: int = 2,
		hyena_filter_order: int = 64,
		ffn_mult: int = 4,
	):
		super().__init__()

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.mixer = HyenaOperator(
			d_model=d_model,
			l_max=l_max,
			order=hyena_order,
			filter_order=hyena_filter_order,
			dropout=dropout,
			filter_dropout=dropout,
		)
		self.ffn = FeedForward(d_model=d_model, dropout=dropout, mult=ffn_mult)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.dropout(self.mixer(self.norm1(x)))
		x = x + self.dropout(self.ffn(self.norm2(x)))
		return x


# ------------------------------------------------------------
# model definition
# ------------------------------------------------------------

class TinyCausalHyenaLM(nn.Module):
	"""
	Tiny causal language model based on stacked Hyena blocks.

	Contract:
	- input: token IDs x of shape (batch_size, seq_len_in)
	- output: logits of shape (batch_size, seq_len_in, vocab_size)
	"""

	def __init__(
		self,
		vocab_size: int,
		seq_len_in: int,
		d_model: int,
		n_layers: int,
		dropout: float,
		hyena_order: int = 2,
		hyena_filter_order: int = 64,
		use_abs_pos: bool = False,
	):
		super().__init__()

		self.vocab_size = int(vocab_size)
		self.seq_len_in = int(seq_len_in)
		self.d_model = int(d_model)
		self.n_layers = int(n_layers)
		self.use_abs_pos = bool(use_abs_pos)

		self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)

		if self.use_abs_pos:
			self.position_embedding = nn.Embedding(self.seq_len_in, self.d_model)
		else:
			self.position_embedding = None

		self.blocks = nn.ModuleList(
			[
				HyenaBlock(
					d_model=self.d_model,
					l_max=self.seq_len_in,
					dropout=dropout,
					hyena_order=hyena_order,
					hyena_filter_order=hyena_filter_order,
					ffn_mult=4,
				)
				for _ in range(self.n_layers)
			]
		)

		self.final_norm = nn.LayerNorm(self.d_model)
		self.lm_head = nn.Linear(self.d_model, self.vocab_size)

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

		hidden = self.token_embedding(x)

		if self.use_abs_pos:
			positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
			hidden = hidden + self.position_embedding(positions)

		for block in self.blocks:
			hidden = block(hidden)

		hidden = self.final_norm(hidden)
		logits = self.lm_head(hidden)
		return logits
