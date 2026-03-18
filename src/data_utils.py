#!/usr/bin/env python3

import random
from pathlib import Path
from typing import Dict, List, Tuple

import ushuffle


VALID_BASES = set("ACGTN")
DNA_BASES = "ACGT"


# ------------------------------------------------------------
# FASTA parsing
# ------------------------------------------------------------

def read_fasta_records(path: Path) -> List[Tuple[str, str]]:
	"""
	Read a multi-FASTA file and return a list of (contig_id, sequence) tuples.

	Behavior:
	- contig_id is the first token after '>'
	- sequence lines are concatenated
	- sequence is uppercased
	- any non-ACGTN character is converted to 'N'
	"""
	records: List[Tuple[str, str]] = []

	current_contig_id = None
	current_seq_parts: List[str] = []

	def flush_record() -> None:
		nonlocal current_contig_id, current_seq_parts

		if current_contig_id is None:
			return

		sequence = "".join(current_seq_parts).upper()
		sequence = "".join(base if base in VALID_BASES else "N" for base in sequence)

		records.append((current_contig_id, sequence))
		current_contig_id = None
		current_seq_parts = []

	with path.open() as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue

			if line.startswith(">"):
				# A new header means the previous record is complete.
				flush_record()
				current_contig_id = line[1:].split()[0]
			else:
				current_seq_parts.append(line)

	# Flush the last record after the loop ends.
	flush_record()
	return records


# ------------------------------------------------------------
# sequence summary helpers
# ------------------------------------------------------------

def gc_fraction(sequence: str) -> float:
	"""
	Compute GC fraction while ignoring Ns.

	GC = (G + C) / (A + C + G + T)

	If there are no A/C/G/T bases, return 0.0.
	"""
	gc_count = sequence.count("G") + sequence.count("C")
	at_count = sequence.count("A") + sequence.count("T")
	denominator = gc_count + at_count

	if denominator == 0:
		return 0.0

	return gc_count / denominator


def n_fraction(sequence: str) -> float:
	"""
	Compute the fraction of Ns in a sequence.
	"""
	if not sequence:
		return 1.0

	return sequence.count("N") / len(sequence)


# ------------------------------------------------------------
# window sampling helpers
# ------------------------------------------------------------

def discover_fna_files(raw_dir: Path) -> List[Path]:
	"""
	Recursively find all .fna files under the raw data directory.
	"""
	return sorted(raw_dir.rglob("*.fna"))


def load_eligible_contigs(raw_dir: Path, window_bp: int) -> Tuple[List[Path], List[Dict[str, str]]]:
	"""
	Load contigs that are long enough to contribute at least one training window.

	The current pipeline uses next-token prediction, so this keeps the original
	pilot threshold of window_bp + 1.
	"""
	fasta_files = discover_fna_files(raw_dir)
	if not fasta_files:
		raise SystemExit(f"No .fna files found under {raw_dir}")

	contigs: List[Dict[str, str]] = []

	for fasta_path in fasta_files:
		genome_id = fasta_path.parent.name
		records = read_fasta_records(fasta_path)

		for contig_id, sequence in records:
			if len(sequence) >= window_bp + 1:
				contigs.append({
					"genome_id": genome_id,
					"contig_id": contig_id,
					"seq": sequence,
				})

	if not contigs:
		raise SystemExit("No contigs long enough for the selected window size.")

	return fasta_files, contigs


def build_contig_sampling_weights(contigs: List[Dict[str, str]], window_bp: int) -> List[int]:
	"""
	Weight each contig by the number of possible window start positions.

	Longer contigs therefore contribute proportionally more candidate windows.
	"""
	weights = [len(contig["seq"]) - window_bp for contig in contigs]

	if any(weight <= 0 for weight in weights):
		raise ValueError("Eligible contigs must have positive sampling weight.")

	return weights


def weighted_choice_index(weights: List[int], rng: random.Random) -> int:
	"""
	Sample one index with probability proportional to its weight.
	"""
	total_weight = sum(weights)
	if total_weight <= 0:
		raise ValueError("Total sampling weight must be positive.")

	draw = rng.randrange(total_weight)
	running_total = 0

	for index, weight in enumerate(weights):
		running_total += weight
		if draw < running_total:
			return index

	return len(weights) - 1


def sample_real_windows(
	contigs: List[Dict[str, str]],
	weights: List[int],
	window_bp: int,
	n_windows: int,
	max_n_frac: float,
	rng: random.Random,
) -> Tuple[List[str], List[Tuple], int, int]:
	"""
	Sample real genomic windows.

	Returns:
		real_windows
		meta_rows
		total_attempts
		rejected_for_n

	The loop continues until the requested number of accepted windows is reached.
	Rejected windows do not count toward that target.
	"""
	real_windows: List[str] = []
	meta_rows: List[Tuple] = []

	total_attempts = 0
	rejected_for_n = 0

	while len(real_windows) < n_windows:
		total_attempts += 1

		# First choose a contig, weighted by how many valid start positions it has.
		contig_index = weighted_choice_index(weights, rng)
		contig = contigs[contig_index]
		sequence = contig["seq"]

		# Then choose one start position within that contig.
		n_start_positions = len(sequence) - window_bp
		start = rng.randrange(n_start_positions)
		end = start + window_bp

		window = sequence[start:end]
		window_n_frac = n_fraction(window)

		# Reject windows with too much ambiguous sequence.
		if window_n_frac > max_n_frac:
			rejected_for_n += 1
			continue

		# Also reject windows with no informative DNA bases at all.
		if (window.count("A") + window.count("C") + window.count("G") + window.count("T")) == 0:
			rejected_for_n += 1
			continue

		window_gc = gc_fraction(window)
		real_windows.append(window)

		meta_rows.append((
			len(real_windows) - 1,
			contig["genome_id"],
			contig["contig_id"],
			start,
			end,
			f"{window_gc:.4f}",
			f"{window_n_frac:.4f}",
		))

	return real_windows, meta_rows, total_attempts, rejected_for_n


def split_indices(n_items: int, val_frac: float, rng: random.Random) -> Tuple[set, set]:
	"""
	Create one shared train/validation split by index.

	The same split is reused for real and both control sets so corresponding
	windows stay aligned across conditions.
	"""
	indices = list(range(n_items))
	rng.shuffle(indices)

	n_val = int(n_items * val_frac)
	val_indices = set(indices[:n_val])
	train_indices = set(indices[n_val:])

	return train_indices, val_indices


# ------------------------------------------------------------
# control generation helpers
# ------------------------------------------------------------

def replace_ns(sequence: str, rng: random.Random) -> str:
	"""
	Replace Ns with random A/C/G/T bases.
	"""
	if "N" not in sequence:
		return sequence

	return "".join(base if base != "N" else rng.choice(DNA_BASES) for base in sequence)


def dinuc_shuffle(sequence: str, rng: random.Random) -> str:
	"""
	Dinucleotide-preserving shuffle using ushuffle.

	Ns are replaced first because ushuffle expects a concrete alphabet.
	"""
	clean_sequence = replace_ns(sequence, rng)
	shuffled_bytes = ushuffle.shuffle(clean_sequence.encode("ascii"), 2)
	return shuffled_bytes.decode("ascii")


def gc_matched_random(sequence: str, rng: random.Random) -> str:
	"""
	Generate an i.i.d. random DNA sequence with matched GC fraction.
	"""
	clean_sequence = replace_ns(sequence, rng)
	gc = gc_fraction(clean_sequence)

	p_a = (1.0 - gc) / 2.0
	p_c = gc / 2.0
	p_g = gc / 2.0
	p_t = (1.0 - gc) / 2.0

	bases = ["A", "C", "G", "T"]
	probabilities = [p_a, p_c, p_g, p_t]

	return "".join(rng.choices(bases, weights=probabilities, k=len(clean_sequence)))


def generate_fixed_controls(real_windows: List[str], rng: random.Random) -> Tuple[List[str], List[str]]:
	"""
	Generate the fixed controls used in this pilot.

	Controls:
	- shuf: dinucleotide-preserving shuffle
	- rand: GC-matched random

	Controls are generated after real-window filtering so each accepted real
	window has one matched shuffled control and one matched random control.
	"""
	shuf_windows: List[str] = []
	rand_windows: List[str] = []

	for real_window in real_windows:
		clean_window = replace_ns(real_window, rng)
		shuf_windows.append(dinuc_shuffle(clean_window, rng))
		rand_windows.append(gc_matched_random(clean_window, rng))

	return shuf_windows, rand_windows


# ------------------------------------------------------------
# file writing helpers
# ------------------------------------------------------------

def write_windows(path: Path, windows: List[str]) -> None:
	"""
	Write one sequence window per line.
	"""
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(windows) + "\n")


def write_meta_tsv(path: Path, meta_rows: List[Tuple]) -> None:
	"""
	Write metadata for the accepted real windows.
	"""
	path.parent.mkdir(parents=True, exist_ok=True)

	with path.open("w") as handle:
		handle.write("idx\tgenome_id\tcontig_id\tstart\tend\tgc\tn_frac\n")
		for row in meta_rows:
			handle.write("\t".join(map(str, row)) + "\n")
