#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ------------------------------------------------------------
# FFT convolution helper
# ------------------------------------------------------------

def fftconv_ref(u: torch.Tensor, k: torch.Tensor, d_term: torch.Tensor) -> torch.Tensor:
	"""
	Reference FFT convolution using padding and truncation to recover linear convolution.

	Shapes:
	- u: (batch_size, channels, seq_len) or broadcast-compatible variant
	- k: (..., channels, seq_len)
	- d_term: (..., channels)
	"""
	seq_len = u.shape[-1]
	fft_size = 2 * seq_len

	k_freq = torch.fft.rfft(k, n=fft_size) / fft_size
	u_freq = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

	if u_freq.ndim > k_freq.ndim:
		while k_freq.ndim < u_freq.ndim:
			k_freq = k_freq.unsqueeze(0)

	y = torch.fft.irfft(u_freq * k_freq, n=fft_size, norm="forward")[..., :seq_len]
	out = y + u * d_term.unsqueeze(-1)
	return out.to(dtype=u.dtype)


# ------------------------------------------------------------
# implicit filter components
# ------------------------------------------------------------

class Sin(nn.Module):
	"""
	Periodic activation used in the implicit Hyena filter MLP.
	"""

	def __init__(self, dim: int, w: float = 1.0, train_freq: bool = True):
		super().__init__()

		if train_freq:
			self.freq = nn.Parameter(torch.full((1, dim), float(w)))
		else:
			self.register_buffer("freq", torch.full((1, dim), float(w)))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.sin(x * self.freq)


class PositionalEmbedding(nn.Module):
	"""
	Complex-exponential positional features used to generate the long filter.
	"""

	def __init__(self, emb_dim: int, seq_len: int):
		super().__init__()

		if emb_dim < 1:
			raise ValueError("emb_dim must be >= 1")

		self.seq_len = int(seq_len)

		t = torch.linspace(0.0, 1.0, self.seq_len)[None, :, None]

		if emb_dim == 1:
			z = t
		else:
			n_bands = (emb_dim - 1) // 2

			if 1 + 2 * n_bands != emb_dim:
				raise ValueError(f"emb_dim must be odd (1 + 2*bands). Got emb_dim={emb_dim}.")

			t_rescaled = torch.linspace(0.0, self.seq_len - 1, self.seq_len)[None, :, None]
			omega = 2 * math.pi * t_rescaled / self.seq_len

			frequencies = torch.linspace(1e-4, max(1e-4, n_bands - 1), n_bands)[None, None]
			z_complex = torch.exp(-1j * frequencies * omega)
			z = torch.cat([t, z_complex.real, z_complex.imag], dim=-1)

		self.register_buffer("z", z)
		self.register_buffer("t", t)

	def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Return positional features z and normalized time t up to seq_len.
		"""
		if seq_len > self.seq_len:
			raise ValueError(f"Requested L={seq_len} > seq_len={self.seq_len} for positional embedding.")

		return self.z[:, :seq_len], self.t[:, :seq_len]


class ExponentialModulation(nn.Module):
	"""
	Apply an exponential decay envelope to the generated long filter.
	"""

	def __init__(
		self,
		d_model: int,
		fast_decay_pct: float = 0.3,
		slow_decay_pct: float = 1.5,
		target: float = 1e-2,
		modulate: bool = True,
		shift: float = 0.05,
		learnable: bool = False,
	):
		super().__init__()

		self.modulate = bool(modulate)
		self.shift = float(shift)

		max_decay = math.log(target) / fast_decay_pct
		min_decay = math.log(target) / slow_decay_pct
		deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]

		if learnable:
			self.deltas = nn.Parameter(deltas)
		else:
			self.register_buffer("deltas", deltas)

	def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			t: Tensor of shape (1, seq_len, 1)
			x: Tensor of shape (1, seq_len, d_model) or (batch_size, seq_len, d_model)
		"""
		if not self.modulate:
			return x

		decay = torch.exp(-t * self.deltas.abs())
		return x * (decay + self.shift)


class HyenaFilter(nn.Module):
	"""
	Implicit long filter generated from positional features with exponential modulation.
	"""

	def __init__(
		self,
		d_model: int,
		emb_dim: int = 3,
		filter_order: int = 64,
		seq_len: int = 2048,
		dropout: float = 0.0,
		w: float = 1.0,
		num_inner_mlps: int = 2,
		modulate: bool = True,
		bias: bool = True,
	):
		super().__init__()

		self.d_model = int(d_model)
		self.seq_len = int(seq_len)
		self.dropout = nn.Dropout(float(dropout))

		self.pos_emb = PositionalEmbedding(emb_dim=emb_dim, seq_len=self.seq_len)
		sin_act = Sin(filter_order, w=w, train_freq=True)

		layers = [nn.Linear(emb_dim, filter_order), sin_act]
		for _ in range(int(num_inner_mlps)):
			layers.append(nn.Linear(filter_order, filter_order))
			layers.append(sin_act)
		layers.append(nn.Linear(filter_order, self.d_model, bias=False))
		self.implicit_filter = nn.Sequential(*layers)

		self.modulation = ExponentialModulation(self.d_model, modulate=modulate)

		if bias:
			self.bias = nn.Parameter(torch.zeros(self.d_model))
		else:
			self.register_buffer("bias", torch.zeros(self.d_model))

	def filter(self, seq_len: int) -> torch.Tensor:
		"""
		Return a generated filter of shape (1, seq_len, d_model).
		"""
		z, t = self.pos_emb(seq_len)
		h = self.implicit_filter(z)
		h = self.modulation(t, h)
		h = self.dropout(h)
		return h

	def forward(
		self,
		x: torch.Tensor,
		seq_len: int,
		k: Optional[torch.Tensor] = None,
		bias: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""
		Apply the generated long convolution.

		Args:
			x: Tensor of shape (batch_size, channels, seq_len)
			k: Optional kernel of shape (channels, seq_len) or (1, seq_len, channels)
			bias: Optional diagonal skip term of shape (channels,)
		"""
		if k is None:
			k = self.filter(seq_len)
		if isinstance(k, tuple):
			k = k[0]

		if k.ndim == 3:
			k = k[0].transpose(0, 1).contiguous()
		elif k.ndim != 2:
			raise ValueError(f"Unexpected kernel shape: {tuple(k.shape)}")

		if bias is None:
			bias = self.bias

		return fftconv_ref(x, k, bias)


# ------------------------------------------------------------
# Hyena operator
# ------------------------------------------------------------

class HyenaOperator(nn.Module):
	"""
	Minimal Hyena sequence mixer.

	Input:
	- u: Tensor of shape (batch_size, seq_len, d_model)

	Output:
	- y: Tensor of shape (batch_size, seq_len, d_model)
	"""

	def __init__(
		self,
		d_model: int,
		l_max: int,
		order: int = 2,
		filter_order: int = 64,
		dropout: float = 0.0,
		filter_dropout: float = 0.0,
		emb_dim: int = 3,
		w: float = 1.0,
	):
		super().__init__()

		if order < 2:
			raise ValueError("HyenaOperator requires order >= 2.")

		self.d_model = int(d_model)
		self.l_max = int(l_max)
		self.order = int(order)

		inner_width = self.d_model * (self.order + 1)

		self.dropout = nn.Dropout(float(dropout))
		self.in_proj = nn.Linear(self.d_model, inner_width)
		self.out_proj = nn.Linear(self.d_model, self.d_model)

		# The short filter handles local mixing before the long filter recurrence.
		self.short_filter = nn.Conv1d(
			in_channels=inner_width,
			out_channels=inner_width,
			kernel_size=3,
			padding=2,
			groups=inner_width,
		)

		self.filter_fn = HyenaFilter(
			d_model=self.d_model * (self.order - 1),
			emb_dim=emb_dim,
			filter_order=filter_order,
			seq_len=self.l_max,
			dropout=filter_dropout,
			w=w,
			num_inner_mlps=2,
			modulate=True,
			bias=True,
		)

	def forward(self, u: torch.Tensor) -> torch.Tensor:
		"""
		Apply Hyena mixing to one sequence batch.
		"""
		_, seq_len, d_model = u.shape

		if d_model != self.d_model:
			raise ValueError(f"Expected last dim D={self.d_model}, got {d_model}")

		filter_len = min(seq_len, self.l_max)

		u = self.in_proj(u)
		u = u.transpose(1, 2).contiguous()
		uc = self.short_filter(u)[..., :filter_len]

		*x_streams, v = uc.split(self.d_model, dim=1)

		k_full = self.filter_fn.filter(filter_len)
		k_full = k_full[0]
		k = k_full.reshape(filter_len, self.order - 1, self.d_model).permute(1, 2, 0).contiguous()
		bias = self.filter_fn.bias.reshape(self.order - 1, self.d_model)

		for order_index, x_i in enumerate(reversed(x_streams[1:])):
			v = self.dropout(v * x_i)
			v = self.filter_fn(v, filter_len, k=k[order_index], bias=bias[order_index])

		y = (v * x_streams[0]).transpose(1, 2).contiguous()
		y = self.out_proj(y)
		return y
